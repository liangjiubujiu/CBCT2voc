# CBCT2voc
This is voc format creation toolkit for .dcm data.
Assuming that you have prepared your original CT data and annotations.
This toolkit can realize:
- [ ]1. generate gray images from each .dcm data.   convert_from_dicom_to_jpg()   generate_gray()
- [ ]2. generate binary mask from annotations including draw contours and filling holes. FillHole()
- [ ]3. rescale the corresponding grayscale image and annotation into a fixed size.
- [ ]4. generate train and val data split .txt according to the input ratio. txt_generate()
```

```
## run
```Shell
python label.py
```
This command helps to create binary masks from gray masks. you will find train.txt, val.txt which are formed by voc data structure.

# Application on deeplab v3+

This is an application on seantic segmentation by [deeplab v3+](https://github.com/liangjiubujiu/pytorch-deeplab-xception).

