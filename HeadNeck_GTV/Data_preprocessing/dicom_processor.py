import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
from DicomRTTool.ReaderWriter import DicomReaderWriter
from rt_utils import RTStructBuilder




# dicom_files = os.listdir(DATA_PATH)

# for dic_file in dicom_files:
# 	dic_file = os.path.join(DATA_PATH,dic_file)
# 	ds = dcmread(dic_file)
# 	print()
# 	print(f"File path........: {dic_file}")
# 	print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
# 	print(ds.dir('contour'))
# 	# print(ds.ROIContourSequence)
# 	ctrs = ds.ROIContourSequence
# 	# print(len(ctrs))
# 	# print(ctrs[0].StructureSetROISequence)
# 	# print(len(ctrs[0].ContourSequence[0].ContourData) )
# 	print(len(ctrs[0].ContourSequence) )
# 	print(len(ctrs[1].ContourSequence) )
# 	print(ctrs[1].ContourSequence)


	# pat_name = ds.PatientName
	# display_name = pat_name.family_name + ", " + pat_name.given_name
	# print(f"Patient's Name...: {display_name}")
	# print(f"Patient ID.......: {ds.PatientID}")
	# print(f"Modality.........: {ds.Modality}")
	# print(f"Study Date.......: {ds.StudyDate}")
	# print(f"Image size.......: {ds.Rows} x {ds.Columns}")
	# print(f"Pixel Spacing....: {ds.PixelSpacing}")
	# print(ds.pixel_array.shape)
	# print(ds.pixel_array.flatten())
	# plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
	# plt.show()

	# break

def main():

	DATA_PATH = '../../../patient_1_label'
	DATA_PATH2 = '../../../patient_1_ct'
	# print(os.path.isfile(os.path.join(DATA_PATH,"RS.1.2.246.352.205.5224286541781341112.5702177400020821633.dcm")))
	# print(os.listdir(DATA_PATH))
	# rtstruct = RTStructBuilder.create_from(
	#   dicom_series_path=DATA_PATH2,
	#   rt_struct_path=os.path.join(DATA_PATH,"RS.1.2.246.352.205.5224286541781341112.5702177400020821633.dcm")
	#   # rt_struct_path=os.path.join(DATA_PATH2,"CT.2.25.33375624801153914503818100773580862655.dcm")
	# )
	# print(rtstruct.get_roi_names())
	# print(vars(rtstruct) )
	# mask_3d = rtstruct.get_roi_mask_by_name("GTVp")
	# mask_3d = rtstruct.get_roi_mask_by_name("BODY")
	# print(mask_3d.shape[-1])
	# l = 0
	# for i in range(mask_3d.shape[-1]):
	# 	first_mask_slice = mask_3d[:, :, i]
	# 	if np.sum(first_mask_slice) != 0:
	# 		plt.imshow(first_mask_slice)
	# 		# plt.show()
	# 		plt.draw()
	# 		if plt.waitforbuttonpress(0) == True:
	# 			plt.close()
	# 			break
	# 		l+=1
	# print("total:{}".format(l))
	# ds = dcmread(os.path.join(DATA_PATH,"RS.1.2.246.352.205.5224286541781341112.5702177400020821633.dcm"))
	ds = dcmread(os.path.join(DATA_PATH,"RS.2.25.3214324626118108117339784454692878167015.dcm"))
	# ds = dcmread(os.path.join(DATA_PATH2,"CT.2.25.33375624801153914503818100773580862655.dcm"))


	try:
		print(ds.StructureSetROISequence)
		for struct_roi in ds.StructureSetROISequence:
			print(struct_roi.ROIName)
	except:
		print("No ROIs")

	# pat_name = ds.PatientName
	# display_name = pat_name.family_name + ", " + pat_name.given_name
	# print(f"Patient's Name...: {display_name}")
	# print(f"Patient ID.......: {ds.PatientID}")
	# print(f"Modality.........: {ds.Modality}")
	# print(f"Study Date.......: {ds.StudyDate}")





def main2():
	DATA_LABEL = '../../../patient_1_label'
	from DicomRTTool.ReaderWriter import DicomReaderWriter

def visualize(images,name='random',i=0):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(9, 9))
    for  image in images:
	# for image in images:
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap = 'bone')
    plt.show()

def main3(DATA_DIR,LABEL_DIR):
	import imageio
	'''
	One can read the IO files
	'''
	rs_path = os.path.join(LABEL_DIR,"RS.2.25.3214324626118108117339784454692878167015.dcm")
	ct_path = os.path.join(DATA_DIR,"CT.2.25.33375624801153914503818100773580862655.dcm")
	# imgs = imageio.mimread(ct_path, 'DICOM')
	# for dcm_file in os.listdir(DATA_DIR):
	# 	res = imageio.mimread(os.path.join(DATA_DIR,dcm_file),'DICOM')
	# 	print(len(res),'\n\n')
	# visualize(imgs)
	# vols = imageio.mvolread("../../../../CT.2.25.33375624801153914503818100773580862655.dcm",'DICOM')
	# print(vols[0].shape,vols[1].shape)
	# ds_base = dcmread("../../../../CT.2.25.33375624801153914503818100773580862655.dcm")
	# plt.imshow(ds_base.pixel_array,cmap=plt.cm.bone)
	# plt.show()

	ds = dcmread(rs_path)
	ds2 = dcmread(os.path.join(LABEL_DIR,"RS.1.2.246.352.205.5224286541781341112.5702177400020821633.dcm"))
	# ds = dcmread(ct_path)


	# for variable in dir(ds):
	# 	print("*******Reading variable******")
	# 	if '__' not in variable:
	# 		try:
	# 			print(variable,getattr(ds,variable))
	# 			print('\n')
	# 		except:
	# 			print("\n\n\n\nSkip variable:{}".format(variable))
	# print(dir(ds.StructureSetROISequence) )

	# print(len(ds.StructureSetROISequence))
	# print(len(ds.ROIContourSequence))
	# print(dir(ds.ROIContourSequence[0]) )

	# for val in dir(ds.StructureSetROISequence[0]):
	# 	print(val)
		# try:
		# 	print('\n\n\n\n\n')
		# 	print("*********************",val,'\n\n',getattr(ds.ROIContourSequence[0],val))
		# except:
		# 	print('\n\n')
	# print(len(ds.ROIContourSequence[0].ContourSequence) )
	# print(len(ds.ROIContourSequence[1].ContourSequence) )
	# print(dir(ds.ROIContourSequence[0].ContourSequence[0]))
	# print(len(ds.ROIContourSequence[1].ContourSequence) )

	# print(len(ds.StructureSetROISequence[1]), ds.StructureSetROISequence[0].ROINumber)
	# for struct_set in ds.StructureSetROISequence:
	# 	print(struct_set.ROIName,struct_set.ROINumber,len(struct_set))

	for roi_contour in ds.ROIContourSequence:
		print(len(roi_contour.ContourSequence))

	series_data = []
	for ct_file in os.listdir(DATA_DIR):
		ds_ct = dcmread(os.path.join(DATA_DIR,ct_file))
		series_data.append(ds_ct)
	series_data.sort(key=lambda ds: ds.SliceLocation, reverse=False)

	uid_list = [series.file_meta.MediaStorageSOPInstanceUID for series in series_data]

	l = 0
	first_list = []
	second_list = []

	for refd_frame_of_ref in ds.ReferencedFrameOfReferenceSequence:
		for rt_refd_study in refd_frame_of_ref.RTReferencedStudySequence:
			for rt_refd_series in rt_refd_study.RTReferencedSeriesSequence:
				for contour_image in rt_refd_series.ContourImageSequence:
					# if os.path.isfile( os.path.join(DATA_DIR,"CT."+contour_image.ReferencedSOPInstanceUID+'.dcm')):
					if contour_image.ReferencedSOPInstanceUID in uid_list:
						first_list.append(contour_image.ReferencedSOPInstanceUID)
						# print("True")
						l += 1
					else:
						raise("Gaga")

	print("Total:{}".format(l))
	l = 0

	for refd_frame_of_ref in ds2.ReferencedFrameOfReferenceSequence:
		for rt_refd_study in refd_frame_of_ref.RTReferencedStudySequence:
			for rt_refd_series in rt_refd_study.RTReferencedSeriesSequence:
				for contour_image in rt_refd_series.ContourImageSequence:
					# if os.path.isfile( os.path.join(DATA_DIR,"CT."+contour_image.ReferencedSOPInstanceUID+'.dcm')):
					if contour_image.ReferencedSOPInstanceUID in uid_list:
						second_list.append(contour_image.ReferencedSOPInstanceUID)
						# print("True")
						l += 1
					else:
						raise("Gaga")

	print("Total:{}".format(l))

	print(set(first_list) & set(second_list))

	# print(len(ds.ROIContourSequence[0].ContourSequence) )
	contour_sequence = ds.ROIContourSequence[0].ContourSequence
	for contour in contour_sequence:
		print(len(contour.ContourImageSequence))
		print(contour.ContourImageSequence[0].ReferencedSOPInstanceUID)


def create_mask_img(DATA_DIR,LABEL_DIR):
	rtstruct  = RTStructBuilder.create_from(dicom_series_path=DATA_DIR, rt_struct_path=LABEL_DIR)
	print(rtstruct.get_roi_names())
	

if __name__ == '__main__':
	DATA_DIR = '../../../patient_1_ct'
	LABEL_DIR = '../../../patient_1_label'
	# main3(DATA_DIR,LABEL_DIR)
	create_mask_img(DATA_DIR,LABEL_DIR)
