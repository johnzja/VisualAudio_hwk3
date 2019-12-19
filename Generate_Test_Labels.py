import numpy as np
import xlrd

file_path = r'D:\iTsinghua\Major\Visual&Audio\hwk\comprehensive_hwk3\dataset\test\labels.xlsx'
# file_path = file_path.decode('utf-8')
data = xlrd.open_workbook(file_path)
table = data.sheet_by_name('Sheet1')

nrows = table.nrows
ncols = table.ncols
col_values = (table.col_values(1))[1:]
np.save('test_labels', col_values)
print(len(col_values))
print('Complete.')
