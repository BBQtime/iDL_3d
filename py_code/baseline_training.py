import global_elems as g
import os
import torch
from tqdm import tqdm
from datetime import datetime
from nested_dict import NestedDict
from torch.utils.data import DataLoader
from shared_training import SharedTraining
from matplotlib import pyplot as plt
from baseline_dataset import BaselineDataSet


class BaselineTraining(SharedTraining):
    def __init__(self):
        super().__init__()

        # record self._epochs_actual because of "early.stop.patience"
        self._epochs_actual = 0
        # self.__tensorboard_writer = TensorBoardWriter(g.BASELINE_TENSORBOARD_FOLDER)

    def _load_hyper(
        self,
        hyper: dict,
        exist_cnn_path: str = None,
        debug_mode: bool = False,  # debug_mode=True will only load 2 epoch and 2 patients
    ):
        # epochs
        if debug_mode:
            self._epochs = 2  # run 2 epochs to compare difference in loss
        else:
            self._epochs = int(hyper["epochs"])
            self._epochs = g.check_limit(self._epochs, 1, None)

        # lr
        self._lr = float(hyper["lr"])
        self._lr = g.check_limit(self._lr, 1e-10, None)
        used_gpu_count = g.used_gpu_count()
        if used_gpu_count > 1:
            self._lr_actual = self._lr * used_gpu_count
        else:
            self._lr_actual = self._lr

        # min lr
        self._lr_min = float(hyper["lr.min"])
        self._lr_min = g.check_limit(self._lr_min, 0.0, self._lr)

        # lr decay patience, based on epoch, must be defined before shared_hyper()
        self._lr_decay_patience = int(hyper["lr.decay.patience"])
        self._lr_decay_patience = g.check_limit(
            self._lr_decay_patience, 1, self._epochs
        )

        # early stop, based on epoch
        self._early_stop_patience = int(hyper["early.stop.patience"])
        self._early_stop_patience = g.check_limit(
            self._early_stop_patience, 1, self._epochs
        )

        # augmentation percent
        self._augment_pct = float(hyper["augment.pct"])
        self._augment_pct = g.check_limit(self._augment_pct, 0.0, 1.0)

        # load shared hyper parameters
        super()._load_hyper(
            hyper=hyper,
            exist_cnn_path=exist_cnn_path,
        )

        # load splitting dataset, based on train/valid/test pct
        # so this must run after shared hyper loaded)
        (
            train_patient_list,
            valid_patient_list,
            test_patient_list,
        ) = self._load_dataset(debug_mode)

        # create dataset
        train_set = BaselineDataSet(
            patient_list=train_patient_list,
            augment_pct=self._augment_pct,
            augment_method=self._augment_method,
            augment_low_limit=self._augment_low_limit,
            augment_up_limit=self._augment_up_limit,
        )
        valid_set = BaselineDataSet(patient_list=valid_patient_list)
        test_set = BaselineDataSet(patient_list=test_patient_list)

        # dataloader
        self._train_loader = DataLoader(
            dataset=train_set,
            batch_size=self._batch_size_actual,
            shuffle=True,  # only shuffle train loader
            num_workers=g.NUM_WORKERS,
        )
        self._valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=self._batch_size_actual,
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )
        self._test_loader = DataLoader(
            dataset=test_set,
            batch_size=self._batch_size_actual,
            shuffle=False,
            num_workers=g.NUM_WORKERS,
        )

    def _print_hyper(self):
        print_dict = NestedDict()
        print_dict["dropout:"] = self._dropout
        print_dict["epochs:"] = self._epochs
        print_dict["early stopping patience:"] = self._early_stop_patience
        print_dict["dataset len:"] = (
            self._train_loader.dataset.__len__()
            + self._valid_loader.dataset.__len__()
            + self._test_loader.dataset.__len__()
        )
        print_dict["train set len:"] = self._train_loader.dataset.__len__()
        print_dict["valid set len:"] = self._valid_loader.dataset.__len__()
        print_dict["test set len:"] = self._test_loader.dataset.__len__()
        super()._print_hyper(print_dict)

    def _save_hyper(self, json_path: str):
        hyper_dict = NestedDict()
        hyper_dict["lr"] = self._lr
        hyper_dict["dataset.len"] = (
            self._train_loader.dataset.__len__()
            + self._valid_loader.dataset.__len__()
            + self._test_loader.dataset.__len__()
        )
        hyper_dict["dataset.train.len"] = self._train_loader.dataset.__len__()
        hyper_dict["dataset.valid.len"] = self._valid_loader.dataset.__len__()
        hyper_dict["dataset.test.len"] = self._test_loader.dataset.__len__()
        hyper_dict["epochs"] = self._epochs
        hyper_dict["epochs.actual"] = self._epochs_actual
        hyper_dict["early.stop.patience"] = self._early_stop_patience
        hyper_dict["dropout"] = self._dropout
        super()._save_hyper(json_path, hyper_dict)

    def loss_fig(self, baseline_id: str):
        loss_json_path = os.path.join(
            g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline", "loss.json"
        )
        self.__loss_fig(loss_json_path)

    def __loss_fig(self, loss_json_path: str):
        plt.figure().clear()

        loss_dict = g.load_json(loss_json_path)
        train_loss = []
        valid_loss = []

        for i in loss_dict:
            train_loss.append(loss_dict[i]["train"])
            valid_loss.append(loss_dict[i]["valid"])

        plt.ylim(min(train_loss) - 0.05, max(train_loss) + 0.05)
        plt.plot(range(1, len(loss_dict) + 1), train_loss, label="train")
        plt.plot(range(1, len(loss_dict) + 1), valid_loss, label="valid")
        plt.legend()

        plt.savefig(loss_json_path[:-4] + "png")

    def __training(self, cnn_save_path: str, loss_save_path: str):
        # g.clear_gpu_cache()
        g.print_line()

        best_epoch = None
        best_loss = None
        patience_count = 0

        for cur_epoch in range(1, self._epochs + 1):
            print("epoch={}".format(cur_epoch))
            print("training:")
            self._cnn.train()
            sum_loss = 0
            batch_num = 0
            for inputs, labels in tqdm(self._train_loader):
                # zero grad at the begining of each mini-batch
                self._optim.zero_grad()
                inputs = inputs.to(g.DEVICE)
                labels = labels.to(g.DEVICE)
                outputs = self._cnn(inputs)
                loss = self._loss_func(outputs, labels)
                loss.backward()  # get grad (must after: optim.zero_grad())
                self._optim.step()  # update param
                sum_loss += loss.item()
                batch_num += 1
            train_loss = sum_loss / batch_num

            # validation
            print("validation:")
            self._cnn.eval()
            with torch.no_grad():
                sum_loss = 0
                batch_num = 0
                for inputs, labels in tqdm(self._valid_loader):
                    inputs = inputs.to(g.DEVICE)
                    labels = labels.to(g.DEVICE)
                    outputs = self._cnn(inputs)
                    loss = self._loss_func(outputs, labels)
                    sum_loss += loss.item()
                    batch_num += 1
            valid_loss = sum_loss / batch_num
            self._scheduler.step(valid_loss)

            # current epoch finished
            self._epochs_actual = cur_epoch

            # save loss in json
            loss_dict = g.load_json(loss_save_path)
            cur_epoch_loss = NestedDict()
            cur_epoch_loss["train"] = train_loss
            cur_epoch_loss["valid"] = valid_loss
            loss_dict["epoch={:03d}".format(self._epochs_actual)] = cur_epoch_loss
            g.save_json(loss_dict, loss_save_path)

            # draw loss figure
            self.__loss_fig(loss_save_path)

            # save cnn
            if best_loss is None:
                best_loss = valid_loss
                best_epoch = cur_epoch
                self._save_cnn(cnn_save_path + str(best_epoch) + ".pt")
            else:
                if valid_loss < best_loss:
                    os.remove(cnn_save_path + str(best_epoch) + ".pt")
                    best_loss = valid_loss
                    best_epoch = cur_epoch
                    self._save_cnn(cnn_save_path + str(best_epoch) + ".pt")
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self._early_stop_patience:
                        break

        # training over
        # g.clear_gpu_cache()

    def training(
        self,
        train_remark: str = None,
        debug_mode: bool = False,
    ):
        for hyper in self._load_group_hyper(g.BASELINE_HYPER_JSON):

            self._load_hyper(
                hyper=hyper,
                debug_mode=debug_mode,
            )
            g.print_line()
            self._print_hyper()

            baseline_id = "baseline_" + self._init_train_id(
                train_remark=train_remark,
                hyper_json_path=g.BASELINE_HYPER_JSON,
                hyper=hyper,
                debug_mode=debug_mode,
            )

            baseline_folder = os.path.join(
                g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline"
            )
            g.create_folder(baseline_folder)

            cnn_save_path = os.path.join(baseline_folder, "epoch=")
            hyper_save_path = os.path.join(baseline_folder, "hyper.json")
            loss_save_path = os.path.join(baseline_folder, "loss.json")

            # save an empty loss.json
            g.save_json(NestedDict(), loss_save_path)

            # save hyper before and after training
            self._save_hyper(hyper_save_path)

            # start training
            self._time_used = datetime.now()
            self.__training(cnn_save_path, loss_save_path)
            self._time_used = datetime.now() - self._time_used
            self._time_used = str(self._time_used).split(".", 2)[0]

            self._save_hyper(hyper_save_path)

            self.inference(
                baseline_id=baseline_id,
                print_hyper=False,
                debug_mode=debug_mode,
            )

    def inference(
        self,
        baseline_id: str,
        print_hyper: bool = True,
        debug_mode: bool = False,
    ):
        g.print_line()
        print(baseline_id)

        baseline_folder = os.path.join(g.TRAIN_RESULTS_FOLDER, baseline_id, "baseline")
        cnn_path = g.get_sub_files(
            baseline_folder, return_full_path=True, key_word=".pt"
        )[0]
        hyper = g.load_json(os.path.join(baseline_folder, "hyper.json"))

        # load and print hyper
        self._load_hyper(
            hyper=hyper,
            exist_cnn_path=cnn_path,
            debug_mode=debug_mode,
        )
        if print_hyper:
            self._print_hyper()

        score = NestedDict()
        for i in g.METRICS_LIST:
            score["avg"][i] = []

        for patient in tqdm(self._test_loader.dataset.patient_list):
            patient_folder = os.path.join(
                baseline_folder, "patients", "patient={}".format(patient)
            )
            g.create_folder(patient_folder)

            # result contains: "gtvs" "dsc" "msc" "hd95"
            patient_result = self._inference_single_patient(patient)

            # save score of cur patient
            for i in g.METRICS_LIST:
                score["patient={}".format(patient)][i] = patient_result[i]
                score["avg"][i].append(patient_result[i])

            # save pred of cur patient
            for i in ["gtvs"]:  # ["gtvt", "gtvn"]:
                g.save_nii(
                    np_data=patient_result[i],
                    save_path=os.path.join(patient_folder, "pred_{}.nii".format(i)),
                    spacing=g.NII_SPACING,
                )
                g.save_nii(
                    np_data=g.binarize_img(patient_result[i]),
                    save_path=os.path.join(
                        patient_folder, "pred_{}_binary.nii".format(i)
                    ),
                    spacing=g.NII_SPACING,
                )

        # save score of all patients
        for i in g.METRICS_LIST:
            avg = score["avg"][i]
            score["avg"][i] = sum(avg) / len(avg)
        g.save_json(
            data=score,
            path=os.path.join(baseline_folder, "score.json"),
        )
