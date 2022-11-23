# import global_elems as g
# import os
# from torch.utils.tensorboard import SummaryWriter


# class TensorBoardWriter:
#     def __init__(self, path: str):
#         self.folder_path = path
#         g.create_folder(self.folder_path)
#         self.clear_cache()
#         self.__writer = SummaryWriter(self.folder_path)

#     def clear_cache(self):
#         file_list = g.get_sub_files(self.folder_path)
#         for file_name in file_list:
#             os.remove(os.path.join(self.folder_path, file_name))
#         return

#     def write_score_per_round(
#         self, idl_id: str, metric_type: str, value: float, cur_round: int
#     ):
#         map_title = metric_type + ".round"
#         scalar_name = idl_id
#         step = cur_round

#         self.__writer.add_scalars(map_title, {scalar_name: value}, step)

#     def write_score_per_iter(
#         self,
#         idl_id: str,
#         metric_type: str,
#         value: float,
#         former_iter_sum: int,
#         cur_iter: int,
#     ):
#         map_title = metric_type + ".iter"
#         scalar_name = idl_id
#         step = former_iter_sum + cur_iter

#         self.__writer.add_scalars(map_title, {scalar_name: value}, step)

#     # for baseline training
#     def write_loss_per_epoch(
#         self,
#         baseline_id: str,
#         train_loss: float,
#         valid_loss: float,
#         epoch: int,
#     ):
#         self.__writer.add_scalars(
#             "loss.epoch",
#             {
#                 baseline_id + "_train": train_loss,
#                 baseline_id + "_valid": valid_loss,
#             },
#             epoch,
#         )

#     def write_loss_per_iter(
#         self,
#         idl_id: str,
#         train_loss: float,
#         _iter: int,
#     ):
#         self.__writer.add_scalars(
#             "loss.iter",
#             {
#                 idl_id + "_train": train_loss,
#             },
#             _iter,
#         )
