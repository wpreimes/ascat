# -*- coding: utf-8 -*-
# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Reader for HSAF SMDAS SWI (ASCAT data assimilation product)
"""
import os
import sys

#if sys.platform.lower() == 'win':
#    os.environ['ECCODES_DEFINITION_PATH'] = r"C:\Users\wpreimes\Anaconda3\envs\ascat\Library\share\eccodes\definitions"

import numpy as np
import pandas as pd
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeogrids.grids import BasicGrid, CellGrid
from pygeogrids.netcdf import load_grid
import pygrib
from typing import Optional, Union
from pygeobase.object_base import Image
from datetime import datetime
import warnings
from pynetcf.time_series import GriddedNcOrthoMultiTs

# todo:
# - time stamps in metadata dont match with timestamps in filename
    # - see e.g. 2017\\h14_20170901_0000.grib has timstamp 2017-09-01 but 201708160000 in metadata
# - Some file are broken after extraction / missing
# - pygeogrids Grid fix is needed to store grid with shape
# - Some files don't contain all 4 SWI variables, e.g. 2017\\h14_20170901_0000.grib

def SMDASGrid(lats: np.ndarray,
              lons: np.ndarray,
              cellsize: Optional[float] = 5.,
              **kwargs) -> Union[CellGrid, BasicGrid]:
    """ Create SMDAS grid from lons, lats read from file. """

    lons_gt_180 = np.where(lons > 180.0)
    lons[lons_gt_180] = lons[lons_gt_180] - 360.

    lons = lons.flatten().astype('float32')
    lats = lats.flatten().astype('float32')

    grid = BasicGrid(lons, lats, **kwargs)

    if cellsize is not None:
        return grid.to_cell_grid(cellsize)
    else:
        return grid


class SMDAS_H14_Img(ImageBase):
    # selection of attributes to search passed parameter name in
    names = ['parameterName', 'name', 'shortName', 'shortNameECMF',
             'nameECMF', 'cfVarName', 'cfVarNameECMF', 'cfName', 'cfNameECMF',
             'paramId', 'paramIdECMF', 'indicatorOfParameter', ]

    # selection of other attributes to read as metadata
    attrs = ['gridType', 'gaussianGridName', 'gridDefinitionDescription',
             'julianDay', 'dataDate', 'units', 'unitsECMF', 'typeOfLevel',
             'typeOfLevelECMF', 'centreDescription',]

    def __init__(self, filename, mode='r',
                 parameters=('swi1', 'swi2', 'swi3', 'swi4'), grid=None,
                 bbox=None, expand_grid=True, ignore_meta=False):
        """

        Parameters
        ----------
        filename
        mode
        parameters : str or List[str], optional (default: swi1-4)
            Parameters to load from files. Case insensitive.
            Can be either short parameterName, name, shortName or shortNameECMF
        grid : None
        bbox : (min_lat, min_lon, max_lat, max_lon)
        expand_grid : bool, optional (default: True)
            Read 2d arrays
        ignore_meta : bool, optional (default: False)
            Do not read metadata attributes from files, slightly faster.
        """

        if (grid is not None) and (bbox is not None):
            raise ValueError("Pass a grid OR a bounding box.")

        super(SMDAS_H14_Img, self).__init__(filename, mode=mode)

        self.params = [parameters] if isinstance(parameters, (str, int)) else parameters
        self.expand_grid = expand_grid
        self.bbox = bbox
        self.shape = None
        self.grid = grid
        self.ignore_meta = ignore_meta

    def _read_img(self, timestamp: Optional[datetime] = None) \
            -> (dict, dict, datetime):
        """ Read data from grib file """

        return_img = {}
        return_metadata = {}

        try:
            with pygrib.open(self.filename) as dataset:
                msgs = [msg for msg in dataset]

        except (IOError, OSError, TypeError) as e:
            print(e)
            raise FileNotFoundError(f"Can not open {self.filename}")

        if self.params is None:
            self.params = [msg['shortName'] for msg in msgs]

        for msg in msgs:

            msg_names = {n: msg[n] for n in self.names}

            p = None
            for p in self.params: # check if current msg contains a param
                if p in msg_names.values():
                    break

            if p is None: continue

            if self.bbox is not None:
                dat, lat, lon = msg.data(*self.bbox)
            else:
                dat, lat, lon = msg.data()

            shape2d = dat.shape

            dat = dat.flatten()

            if self.grid is None:
                self.grid = SMDASGrid(lat, lon, cellsize=5., shape=shape2d)
            else:
                assert len(self.grid.activegpis) == np.product(self.grid.shape), \
                    "Number of grid points does not match with data"

            return_img[p] = dat[self.grid.activegpis]

            if self.ignore_meta:
                meta = {}
            else:
                meta = msg_names.copy()

                try:
                    meta['_FillValue'] = msg.values.fill_value
                except AttributeError:
                    meta['_FillValue'] = None

                for a in self.attrs:
                    try:
                        meta[a] = msg[a]
                    except RuntimeError:
                        meta[a] = None

            dt = datetime(msg['year'], msg['month'], msg['day'])

            if timestamp is None:
                timestamp = dt
            else:
                if dt != timestamp:
                    warnings.warn(f"Passed timestamp does not match with file:"
                                  f" {timestamp} and {dt}")

            return_metadata[p] = meta

        return return_img, return_metadata, timestamp

    def read(self, timestamp: Optional[datetime] = None) -> Image:
        """ Read single HSAF SMDAS image in grib format """

        try:
            return_img, return_metadata, timestamp = self._read_img(timestamp)

            img_params = return_img.keys()

            # some files are missing some params ... e.g. h14_20170901_0000.grib
            if any([p not in img_params for p in self.params]):
                return_img_empty, return_metadata_empty, _ = self._read_empty(timestamp)

                for p in self.params:
                    if p not in img_params:
                        warnings.warn(f"No variable {p} found in existing file,"
                                      f" replaced with empty image")
                        return_img[p] = return_img_empty[p]
                        return_metadata[p] = return_metadata_empty[p]

        except (IOError, FileNotFoundError):
            return_img, return_metadata, timestamp = self._read_empty(timestamp)
            warnings.warn(f"No file {self.filename} found, replaced with empty image")


        if not self.expand_grid:
            return Image(lon=self.grid.arrlon,
                         lat=self.grid.arrlat,
                         data=return_img,
                         metadata=return_metadata,
                         timestamp=timestamp)
        else:
            for key in return_img:
                return_img[key] = np.flipud(
                    return_img[key].reshape(self.grid.shape))

            return Image(lon=self.grid.arrlon,
                         lat=np.flipud(self.grid.activearrlat.reshape(self.grid.shape)),
                         data=return_img,
                         metadata=return_metadata,
                         timestamp=timestamp)

    def _read_empty(self, timestamp, fill_value:float=9999.) -> (dict, dict):
        """ Create an empty image (all nans) """
        
        if self.params is None:
            raise ValueError("No parameters defined to create empty image for.")

        if self.grid is None:
            raise ValueError("No grid defined to create empty image from.")

        return_img, return_meta = {}, {}

        for param in self.params:
            data = np.ma.masked_array(np.full(self.grid.shape, np.nan),
                                      mask=True, fill_value=fill_value)
            return_img[param] = data.flatten()
            if not self.ignore_meta:
                return_meta[param] = {'image_missing': True, '_FillValue': fill_value}

        return return_img, return_meta, timestamp


    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class SMDAS_H14_Ds(MultiTemporalImageBase):
    """
    Class for reading HSAF SMDAS images in grib format as data stacks.

    Parameters
    ----------
    data_path : str
        Path to the grib files
    img_kwargs :
        Additional kwargs are used to initialise the SMDAS Image
    """

    sub_path = ['%Y']
    filename_templ = "*{datetime}*.grib"

    def __init__(self, data_path, parameters=('swi1', 'swi2', 'swi3', 'swi4'),
                 **img_kwargs):
        
        self.img_kwargs = img_kwargs

        try:
            self.grid = img_kwargs['grid']
        except KeyError:
            self.grid = None
        
        self.img_kwargs['parameters'] = parameters
        self.img_kwargs['grid'] = self.grid

        super(SMDAS_H14_Ds, self).__init__(data_path,
                                           SMDAS_H14_Img,
                                           fname_templ=self.filename_templ,
                                           datetime_format="%Y%m%d",
                                           subpath_templ=self.sub_path,
                                           exact_templ=False,
                                           ioclass_kws=self.img_kwargs)

    def _assemble_img(self, timestamp, mask=False, **kwargs):
        img = None
        try:
            filepath = self._build_filename(timestamp)
        except IOError:
            filepath = None

        if self._open(filepath):
            kwargs['timestamp'] = timestamp
            if self.fid.grid is None:
                self.fid.grid = self.grid
            if mask is False:
                img = self.fid.read(**kwargs)
            else:
                img = self.fid.read_masked_data(**kwargs)

        return img

    def read(self, timestamp, **kwargs):
        print(f"Read {timestamp}")
        img =  self._assemble_img(timestamp, **kwargs)
        if self.grid is None:
            self.grid = self.fid.grid
        return img

    def tstamps_for_daterange(self,
                              start_date: datetime,
                              end_date: datetime) -> np.array:
        """
        Create daily time stamps for the passed date range
        """
        return pd.date_range(start_date, end_date, freq='D').to_pydatetime()

    def reshuffle(self, out_path, startdate, enddate, imgbuffer=500,
                  global_attrs=None, var_attrs=None):
        try:
            from repurpose.img2ts import Img2Ts
        except ImportError:
            raise ImportError("Time series conversion needs 'repurpose' package. "
                              "Install it with 'pip install repurpose' first")

        if var_attrs is None:
            var_attrs = {}

        img_kwargs = self.img_kwargs

        if img_kwargs['parameters'] is None:
            raise ValueError('None is not a valid input for parameters during reshuffling.')

        if 'expand_grid' in img_kwargs:
            if img_kwargs['expand_grid']:
                warnings.warn("Reshuffling needs 1d-array, turing grid expanding off.")
                img_kwargs['expand_grid'] = False
        else:
            img_kwargs['expand_grid'] = False

        self.img_kwargs = img_kwargs

        path_firstfile = self._build_filename(startdate)

        if not os.path.isfile(path_firstfile):
            raise FileNotFoundError(f"File for start date must exist, but "
                                    f"{path_firstfile} not found.")

        img1_reader = self.ioclass(path_firstfile, self.img_kwargs)
        img1 = img1_reader.read()  # this is done to set the grid from data in the object

        if self.grid is None:
            self.grid = img1_reader.grid

        # in_grid = self.grid
        #
        # if len(in_grid.shape) ==2: # hack to fix pygeogrids bug, see issue #65
        #     in_grid = (np.product(in_grid.shape),)

        var_attrs.update(img1.metadata)

        # todo: THIS NEEDS THE PYGEOGRIDS METADATA BRANCH INSTALLED WITH FIX FOR SHAPE
        reshuffler = Img2Ts(input_dataset=self, outputpath=out_path,
                            startdate=startdate, enddate=enddate, input_grid=self.grid,
                            imgbuffer=imgbuffer, cellsize_lat=5.0,
                            cellsize_lon=5.0, global_attr=global_attrs, zlib=True,
                            unlim_chunksize=1000, ts_attributes=var_attrs)
        reshuffler.calc()


class SMDASTs(GriddedNcOrthoMultiTs):
    def __init__(self, ts_path, grid_path=None, **kwargs):
        """
        Class for reading GLDAS time series after reshuffling.
        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional (default: None)
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.
        Optional keyword arguments that are passed to the Gridded Base:
        ------------------------------------------------------------------------
            parameters : list, optional (default: None)
                Specific variable names to read, if None are selected, all are read.
            offsets : dict, optional (default:None)
                Offsets (values) that are added to the parameters (keys)
            scale_factors : dict, optional (default:None)
                Offset (value) that the parameters (key) is multiplied with
            ioclass_kws: dict
                Optional keyword arguments to pass to OrthoMultiTs class:
                ----------------------------------------------------------------
                    read_bulk : boolean, optional (default:False)
                        if set to True the data of all locations is read into memory,
                        and subsequent calls to read_ts read from the cache and not from disk
                        this makes reading complete files faster#
                    read_dates : boolean, optional (default:False)
                        if false dates will not be read automatically but only on specific
                        request useable for bulk reading because currently the netCDF
                        num2date routine is very slow for big datasets
        """
        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path)
        super(SMDASTs, self).__init__(ts_path, grid, **kwargs)


if __name__ == '__main__':
    from datetime import datetime


#     img_reader = SMDAS_H14_Img(os.path.join(path, '2015', 'h14_20150106_0000.grib'))
#     img1 = img_reader.read()
#     grid = img_reader.grid
# #    img1 = img_reader.read(datetime(2015,1,6))
#
#     img_reader = SMDAS_H14_Img(os.path.join(path, '2015', 'h14_20160106_0000.grib'),
#                                grid=grid)
#     img2 = img_reader.read(datetime(2016,1,6))

    path = r"R:\Projects\SMART-DRI\07_data\SMDAS2_H14\refurbished"
    out_path = r"C:\Temp\smdas_Ts"

    ds = SMDAS_H14_Ds(path)
    ds.reshuffle(out_path=out_path, startdate=datetime(2017, 8, 29),
                 enddate=datetime(2017, 9, 2), imgbuffer=300)
