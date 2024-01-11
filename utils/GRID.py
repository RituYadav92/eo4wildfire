import gdal
class GRID:
  # read image files
  def read_data(filename):
    raster = gdal.Open(filename)  # open file

    im_width = raster.RasterXSize  # get width
    im_height = raster.RasterYSize  # get height

    im_geotrans = raster.GetGeoTransform()  # get geoTransform
    im_proj = raster.GetProjection()  # get Projection
    im_data = raster.ReadAsArray(0, 0, im_width, im_height)  # read data as array
    mask = raster.GetRasterBand(1).GetMaskBand().ReadAsArray(0, 0)

    del raster
    return im_proj, im_geotrans, im_data, mask

  # write tiff file
  def write_data(filename, im_proj, im_geotrans, im_data, bandNameList):
    # gdal data types include:
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # check the datatype of raster data
    if 'int8' in im_data.dtype.name:
      datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
      datatype = gdal.GDT_UInt16
    else:
      datatype = gdal.GDT_Float32

    # get the dimension
    if len(im_data.shape) == 3:
      im_bands, im_height, im_width = im_data.shape
    else:
      im_bands, (im_height, im_width) = 1, im_data.shape

    # create the output file
    driver = gdal.GetDriverByName("GTiff")  # specify the format
    raster = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if (raster != None):
      raster.SetGeoTransform(im_geotrans)  # write affine transformation parameter
      raster.SetProjection(im_proj)  # write Projection
    else:
      print("Fails to create output file !!!")

    if im_bands == 1:
      rasterBand = raster.GetRasterBand(1)
      rasterBand.SetNoDataValue(0)

      rasterBand.SetDescription(bandNameList[0])
      rasterBand.WriteArray(im_data)

    else:
        # print("im_bands", im_bands)
        for bandIdx, bandName in zip(range(0, im_bands), bandNameList):
          # print(bandIdx)
          # print(bandName)
          bandNum = bandIdx + 1
          rasterBand = raster.GetRasterBand(bandNum)
          rasterBand.SetNoDataValue(0)

          rasterBand.SetDescription(bandName)
          rasterBand.WriteArray(im_data[bandIdx, ...])

    del raster

#################################
run = GRID