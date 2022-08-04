import os
import shutil
import numpy as np
import warnings
import imageio
import imageio.core.util
import SimpleITK as sitk


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning
warnings.filterwarnings("ignore")


class NiftiSlicer:
    # min value=0, max value=1, dtype=float32
    def __nii_data_normalize(self):
        # make min value=0
        self.__nii_data.ct = self.__nii_data.ct - self.__nii_data.ct.min()
        self.__nii_data.pet = self.__nii_data.pet - self.__nii_data.pet.min()
        self.__nii_data.mrt1 = self.__nii_data.mrt1 - self.__nii_data.mrt1.min()
        self.__nii_data.mrt2 = self.__nii_data.mrt2 - self.__nii_data.mrt2.min()
        # dtype = float32 #
        self.__nii_data.ct = np.array(self.__nii_data.ct, dtype=np.float32)
        self.__nii_data.pet = np.array(self.__nii_data.pet, dtype=np.float32)
        self.__nii_data.mrt1 = np.array(self.__nii_data.mrt1, dtype=np.float32)
        self.__nii_data.mrt2 = np.array(self.__nii_data.mrt2, dtype=np.float32)
        self.__nii_data.label = np.array(self.__nii_data.label, dtype=np.float32)
        # make range between [0-1]
        self.__nii_data.ct /= self.__nii_data.ct.max()
        self.__nii_data.pet /= self.__nii_data.pet.max()
        self.__nii_data.mrt1 /= self.__nii_data.mrt1.max()
        self.__nii_data.mrt2 /= self.__nii_data.mrt2.max()

    # Save multi-cnn Nifti image names
    class __MultiModelNiiNames:
        def __init__(self):
            self.ct = ""
            self.pet = ""
            self.mrt1 = ""
            self.mrt2 = ""
            self.label = ""

    # Save multi-cnn Nifti image data
    class __MultiModelNiiData:
        def __init__(self):
            # These variables must be initialized
            self.ct = np.zeros([1], dtype=np.float32)
            self.pet = np.zeros([1], dtype=np.float32)
            self.mrt1 = np.zeros([1], dtype=np.float32)
            self.mrt2 = np.zeros([1], dtype=np.float32)
            self.label = np.zeros([1], dtype=np.float32)
            self.depth = 0

    def __get_patiend_id_from_nii_name(self, nii_name):
        patient_id = "".join(i for i in nii_name if i.isdigit())
        # get left 3 characters, because there is number '1' in 'mtr1'
        patient_id = patient_id[0:3]
        return patient_id

    #  Get multi-model Nifti image names using patient ID #
    def __load_multi_model_nii_names(self):
        # must initialize nii_names at first
        self.__nii_names = self.__MultiModelNiiNames()
        if self.__gtv_based == "gtvt":
            gtv_str = "_GTVt.nii"
        elif self.__gtv_based == "gtvn":
            gtv_str = "_GTVn.nii"
        elif self.__gtv_based == "gtvs":
            gtv_str = "_GTVs.nii"
        for nii_name in self.__nii_name_list:
            patient_id = self.__get_patiend_id_from_nii_name(nii_name)
            if patient_id == self.__cur_patient_id:
                if "_CT.nii" in nii_name:
                    self.__nii_names.ct = nii_name
                elif "_PT.nii" in nii_name:
                    self.__nii_names.pet = nii_name
                elif "_T1dr.nii" in nii_name:
                    self.__nii_names.mrt1 = nii_name
                elif "_T2dr.nii" in nii_name:
                    self.__nii_names.mrt2 = nii_name
                elif gtv_str in nii_name:
                    self.__nii_names.label = nii_name
                self.__nii_name_list_to_remove.remove(nii_name)
            # find next patient, exit loop to save some time
            elif int(patient_id) > int(self.__cur_patient_id):
                break
        self.__nii_name_list = self.__nii_name_list_to_remove.copy()

    def __load_np_from_nii(self, file_path):
        read_img = sitk.ReadImage(file_path)
        np_img = sitk.GetArrayFromImage(read_img)
        return np_img

    def __check_nii_shape(self, input_shape):
        assert (
            input_shape == self.__nii_data.pet.shape
        ), "__load_multi_model_nii_data(): nifti data have different shape: {}".format(
            input_shape
        )

    def __load_multi_model_nii_data(self):
        self.__nii_data = self.__MultiModelNiiData()
        # Load PET data first, because there is always PET image #
        self.__nii_data.pet = self.__load_np_from_nii(
            os.path.join(self.NII_FOLDER_PATH, self.__nii_names.pet)
        )
        self.__nii_data.depth = self.__nii_data.pet.shape[0]
        # Load CT data #
        if self.__nii_names.ct != "":
            self.__nii_data.ct = self.__load_np_from_nii(
                os.path.join(self.NII_FOLDER_PATH, self.__nii_names.ct)
            )
            self.__check_nii_shape(self.__nii_data.ct.shape)
        # Load MRt1 data #
        if self.__nii_names.mrt1 != "":
            self.__nii_data.mrt1 = self.__load_np_from_nii(
                os.path.join(self.NII_FOLDER_PATH, self.__nii_names.mrt1)
            )
            self.__check_nii_shape(self.__nii_data.mrt1.shape)
        # Load MRt2 data #
        if self.__nii_names.mrt2 != "":
            self.__nii_data.mrt2 = self.__load_np_from_nii(
                os.path.join(self.NII_FOLDER_PATH, self.__nii_names.mrt2)
            )
            self.__check_nii_shape(self.__nii_data.mrt2.shape)
        # Load Label data #
        if self.__nii_names.label != "":
            self.__nii_data.label = self.__load_np_from_nii(
                os.path.join(self.NII_FOLDER_PATH, self.__nii_names.label)
            )
            self.__check_nii_shape(self.__nii_data.label.shape)
        # Data Normalization #
        self.__nii_data_normalize()

    def __create_cur_patient_folder(self):
        self.__cur_patient_folder_path = ""
        self.__cur_patient_folder_path = (
            os.path.join(self.SLICES_FOLDER_PATH, self.__cur_patient_id) + "/"
        )
        # empty folder
        if os.path.exists(self.__cur_patient_folder_path):
            shutil.rmtree(self.__cur_patient_folder_path)
        # create folder
        os.makedirs(self.__cur_patient_folder_path)

    # Save Single Slice (PatientID_Model_SliceNr.png)=========
    def __save_single_slice(self, slice_data, slice_id, file_name):
        # 'slice_id' is int, change it into a string start with '0'
        slice_id = str(slice_id).zfill(3)
        save_path = os.path.join(self.__cur_patient_folder_path, slice_id)
        if os.path.exists(save_path) is False:  # create path if doesnt exist
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file_name)
        # save numpy #
        np.save(save_path + ".npy", slice_data)
        # save png #
        slice_data *= 255  # must do this before save PNG format
        imageio.imwrite(save_path + ".png", slice_data)

    def __save_multi_model_slices(self, slice_id):
        # Get Slices #
        if self.__nii_names.ct != "":
            ct_slice = self.__nii_data.ct[slice_id]
        if self.__nii_names.pet != "":
            pet_slice = self.__nii_data.pet[slice_id]
        if self.__nii_names.mrt1 != "":
            mrt1_slice = self.__nii_data.mrt1[slice_id]
        if self.__nii_names.mrt2 != "":
            mrt2_slice = self.__nii_data.mrt2[slice_id]
        if self.__nii_names.label != "":
            label_slice = self.__nii_data.label[slice_id]
        # Save slices #
        self.__save_single_slice(ct_slice, slice_id, "ct")
        self.__save_single_slice(pet_slice, slice_id, "pet")
        self.__save_single_slice(mrt1_slice, slice_id, "mrt1")
        self.__save_single_slice(mrt2_slice, slice_id, "mrt2")
        self.__save_single_slice(label_slice, slice_id, "label")

    def __extend_tumor_slices_id_list(self):
        if self.__extend_pct <= 0.0:
            return
        if self.__nii_names.label == "":
            return
        extend_len = int(self.__extend_pct * len(self.__tumor_slices_id_list))
        list_head = self.__tumor_slices_id_list[0]
        list_tail = self.__tumor_slices_id_list[-1]
        head_id = self.__all_slices_id_list.index(list_head)
        tail_id = self.__all_slices_id_list.index(list_tail)
        for i in range(extend_len):
            if (tail_id + 1) < len(self.__all_slices_id_list):
                self.__tumor_slices_id_list.append(
                    self.__all_slices_id_list[tail_id + 1]
                )
                tail_id += 1
        for i in range(extend_len):
            if head_id > 0:
                self.__tumor_slices_id_list.insert(
                    0, self.__all_slices_id_list[head_id - 1]
                )
                head_id -= 1

    def __get_slices_id_lists(self):
        self.__all_slices_id_list = []
        self.__tumor_slices_id_list = []
        for slice_id in range(self.__nii_data.depth):
            if self.__nii_names.label != "":
                self.__all_slices_id_list.append(slice_id)
                if self.__tumor_only:
                    label_slice = self.__nii_data.label[slice_id]
                    if label_slice.max() != label_slice.min():  # find tumor
                        self.__tumor_slices_id_list.append(slice_id)
                else:  # all slices
                    self.__tumor_slices_id_list.append(slice_id)

    def __save_all_slices(self):
        if self.__nii_names.label == "":
            return
        self.__create_cur_patient_folder()
        for slice_id in self.__tumor_slices_id_list:
            self.__save_multi_model_slices(slice_id)

    def __slice_cur_patient_nii(self):
        # load nii names and data
        self.__load_multi_model_nii_names()
        self.__load_multi_model_nii_data()
        # get slices id lists
        self.__get_slices_id_lists()
        self.__extend_tumor_slices_id_list()
        # save slices
        self.__save_all_slices()

    def __init_path(self):
        self.NII_FOLDER_PATH = "F:/alan/Scans3mm/"
        self.SLICES_FOLDER_PATH = "F:/alan/dataset_2d/"
        self.SLICES_FOLDER_PATH = os.path.join(
            self.SLICES_FOLDER_PATH, self.__gtv_based
        )
        if not self.__tumor_only:
            folder_name = "3mm_all_slices/"
        elif self.__extend_pct == 0.0:
            folder_name = "3mm_tumor_only/"
        else:
            folder_name = "3mm_tumor_extend_pct={}/".format(self.__extend_pct)
        self.SLICES_FOLDER_PATH = os.path.join(self.SLICES_FOLDER_PATH, folder_name)

    def run(self, gtv_based, tumor_only, extend_pct=0.0):
        self.__gtv_based = gtv_based
        if extend_pct > 1.0:
            extend_pct = 1.0
        if extend_pct < 0.0:
            extend_pct = 0.0
        self.__tumor_only = tumor_only
        self.__extend_pct = extend_pct
        self.__init_path()
        self.__nii_name_list = os.listdir(self.NII_FOLDER_PATH)
        self.__nii_name_list.sort()
        # create  [nii_name_list_to_remove] because it's not safe to
        # remove an element from current activated list in for loop
        self.__nii_name_list_to_remove = self.__nii_name_list.copy()
        print("nii_name_list:", len(self.__nii_name_list))
        print("nii_name_list_to_remove:", len(self.__nii_name_list_to_remove))
        # loop all nii files in the folder #
        while len(self.__nii_name_list) > 0:
            self.__cur_patient_id = self.__get_patiend_id_from_nii_name(
                self.__nii_name_list[0]
            )
            print("current patient id:", self.__cur_patient_id)
            self.__slice_cur_patient_nii()
            if 0:
                break
        print("nii_name_list:", len(self.__nii_name_list))
        print("nii_name_list_to_remove:", len(self.__nii_name_list_to_remove))


# main #
if __name__ == "__main__":
    slicer = NiftiSlicer()
    slicer.run(gtv_based="gtvs", tumor_only=True, extend_pct=0.0)  # also "gtvt" "gtvt"
    print("================ End Line ================")
