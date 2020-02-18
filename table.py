#!/usr/bin/env python
__author__ = "David Dennis"
__maintainer__ = "David Dennis"

"""
 * Original Creation Date: Sept 18, 2016
 * Python Version: 3.6.1
 * Instructions: 
 *     Please do not use this software code without my express permission.
 *     That said, non-profit experimentation is welcome.
"""

import numpy as np
from numpy import int64
import pandas as pd
from pandas import DataFrame, Series, RangeIndex
from tabulate import tabulate
from util import apply_op


class Table(DataFrame):
    @property
    def num_rows(self):
        return self.shape[0]

    @property
    def num_columns(self):
        return self.shape[1]

    def adv_string_where(self, column_name, or_phrases, join='and'):
        mask = self[column_name].str.contains(or_phrases[0])
        if len(or_phrases) > 1:
            for phrase in or_phrases[1:]:
                if join == 'and':
                    mask &= self[column_name].str.contains(phrase)
                elif join == 'or':
                    mask |= self[column_name].str.contains(phrase)
                else:
                    raise NotImplementedError
        df = self[mask]
        df.__class__ = Table
        return df

    def string_where(self, column_name, search_str):
        df = self[self[column_name].str.contains(search_str)]
        df.__class__ = Table
        return df

    def col_stats(self):
        return [self.iloc[:,i].apply(type).value_counts() for i in range(self.shape[1])]

    def append_row(self, row_list, ignore_index=True):
        """
        Note: Returns a new DF
        """
        row_series = Series(row_list, index=self.columns)
        df = self.append(row_series, ignore_index=ignore_index)
        df.__class__ = Table
        return df        

    def get_subset(self, limit):
        """
        Note: Returns a new DF
        """
        df = self.ix[:limit]
        df.__class__ = Table
        return df

    def increment_val(self, column_name, index_val):
        val = int(self.get_cell_by_index(column_name, index_val))
        val += 1
        self.set_cell_value(column_name, index_val, val)

    def get_column_name(self, ident):
        ident = self.format_ident(ident)
        if isinstance(ident, int):        
            return list(self.columns)[ident]
        else:
            raise NotImplementedError

    def get_column_names(self, idents=None):
        if idents is None:
            return list(self.columns)
        return [self.get_column_name(ident) for ident in idents]
            
    def get_column(self, ident):
        ident = self.format_ident(ident)
        if isinstance(ident, int):
            return self[self.columns[ident]]
        elif isinstance(ident, str):
            return self[ident]
        else:
            raise NotImplementedError

    def get_columns(self, idents):
        return [self.get_column(ident) for ident in idents]

    def exclude_columns(self, idents):
        """
            Note: Returns a new df
        """
        new_tbl = self.drop(idents, axis=1)
        new_tbl.__class__ = Table
        return new_tbl

    def cast_columns(self, idents, typ):
        if not (isinstance(idents, list) or isinstance(idents, tuple)):
            raise NotImplementedError
        for ident in idents:
            self.cast_column(ident, typ)

    def cast_column(self, ident, typ):
        """
            Note: Returns a set-able Column
            Eg., row[x] = df.apply_rows(lambda x:)
        """        
        ident = self.format_ident(ident)
        return self[ident].astype(typ)

    def apply_columns(self, idents, mod_gen):
        for ident in idents:
            self.apply_column(ident, mod_gen)

    def apply_column(self, ident, mod_gen):
        """
            Only self-applies to target column
            Note: mod_gen can be a row lambda
        """
        ident = self.format_ident(ident)
        target_series = self.get_column(ident)
        if isinstance(ident, int):
            self.iloc[:, ident] = target_series.apply(mod_gen)
        elif isinstance(ident, str):
            self[ident] = target_series.apply(mod_gen)

    def apply_rows(self, row_fn):
        """
            Note: Returns a set-able Column
            Eg., df[name] = df.apply_rows(lambda row:)
        """
        return self.apply(lambda row: row_fn(row), axis=1)

    def easy_set_index(self, ident, drop=True):
        """
        Note: Returns a new DF
        """
        ident = self.format_ident(ident)
        col = self.get_column(ident)
        df = self.set_index(col)
        df.__class__ = Table
        if drop:
            df.drop_column(ident)
        return df

    def copy_index_to_new_col(self, column_name=None):
        self.reset_index(inplace=True)
        if column_name:
            self.rename_column('index', column_name)

    def get_row_by_index(self, index_val):
        return self.iloc[index_val]

    def get_cell_by_index(self, column_name, index_val):
        index_val = self.format_ident(index_val)
        val = self.ix[index_val,][column_name]
        return val

    def set_cell_value(self, column_name, index_val, new_val):
        self.at[index_val, column_name] = new_val

    ### FILTERING
    def filter_by_value(self, column_name, op, val):
        return self[apply_op(self[column_name], op, val)]

    def filter_and_sum(self, column_name, op, value, column_name2):
        return self.filter_by_value(column_name, op, value)[column_name2].sum()

    def filter_rows_by_col_val(self, ident, value_s, exclude=False):
        ident = self.format_ident(ident)
        values = value_s if isinstance(value_s, list) else [value_s]
        if isinstance(ident, str):
            if exclude:
                self.drop(self[self[ident].isin(values)].index,
                          inplace=True)                
            else:
                self.drop(self[~self[ident].isin(values)].index,
                          inplace=True)
        elif isinstance(ident, int):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def filter_columns(self, column_names):
        df = self[column_names]
        df.__class__ = Table
        return df

    def normalize_column(self, col_name):
        try:
            max_val = self.max_column(col_name)
        except TypeError:
            max_val = max(self[col_name].values)
        new_col_name = '%s_norm' % Table.format_col_name(col_name)
        self.add_column(new_col_name, typ=float)
        self[new_col_name] = self.apply_rows(lambda row: row[col_name]/max_val)

    @classmethod
    def format_col_name(cls, col_name):
        return col_name.lower().replace(' ', '_')

    def append_column(self, series):
        df = pd.concat([self, series.to_frame().T])
        df.__class__ = Table
        return df

    def add_column(self, *args, **kwargs):
        typ = kwargs.get('typ')
        default = kwargs.get('default')
        empty = kwargs.get('empty')
        if len(args) == 1:
            column_name = args[0]
            if empty:
                self[column_name] = ''
            elif default:
                self[column_name] = default                
            else:
                if typ in (int, float,):
                    self[column_name] = Series(np.zeros(len(self)),
                                               index=self.index,
                                               dtype=typ)
                elif typ in (str,) or typ is None:
                    self[column_name] = Series('',
                                               index=self.index,
                                               dtype=typ)
        if len(args) == 2:
            column_name, gen = args
            self[column_name] = Series(dtype=typ)
            self[column_name] = self.apply(gen, axis=1)

    def add_columns(self, column_names, typ=None, empty=False):
        for column_name in column_names:
            self.add_column(column_name, typ=typ, empty=empty)

    def drop_columns(self, column_names):
        for column_name in column_names:
            self.drop_column(column_name)

    def drop_column(self, ident):
        if isinstance(ident, str):
            column_name = ident
            self.drop(column_name, axis=1, inplace=True)
        elif isinstance(ident, int):
            self.drop(self.columns[ident], axis=1, inplace=True)
        else:
            raise NotImplementedError

    def drop_rows_by_index(self, index_vals):
        """
        Note: Returns a new DF
        """
        index_vals = [index_vals] if not isinstance(index_vals, list) else index_vals
        df = self[~self.index.isin(index_vals)]
        df.__class__ = Table
        return df        

    def drop_rows_by_value(self, *args, **kwargs):
        """
        Note: Returns a new DF
        """
        exclude = kwargs.get('exclude')
        column_name, values = args
        if not isinstance(values, list):
            values = [values]
        if exclude:
            df = self[self[column_name].isin(values)]            
        else:
            df = self[~self[column_name].isin(values)]
        df.__class__ = Table
        return df

    def remove_duplicates(self, column_name):
        return self.drop_duplicates(subset=column_name, inplace=True)

    def sort_by(self, ident, desc=True):
        ident = self.format_ident(ident)
        ascending = not desc
        if isinstance(ident, str):
            self.sort_values(by=ident, inplace=True, ascending=ascending)
        else:
            raise NotImplementedError        

    def rename_column_by_id(self, ident, new_name):
        old_col_name = self.get_column_name(ident)
        self.rename_column(old_col_name, new_name)

    def rename_column(self, orig_name, new_name):
        self.rename(columns={orig_name: new_name},
                    inplace=True)

    def rename_columns(self, new_name_list):
        if len(new_name_list) != len(self.columns):
            raise Exception('Column size must match')
        self.columns = new_name_list

    def apply_column_names(self, mod_gen):
        """
        Note: Must set return value to `df.columns = `
        """
        return self.columns.map(mod_gen)

    def count_where(self, *args):
        """
        Note: Only handles one condition, for now..
        """
        if all([isinstance(a, str) for a in args]):
            x, op, y = args
            return len(self.filter_by_value(x, op, y))

    def max_column(self, column_name):
        return self.iloc[self[column_name].idxmax()][column_name]

    def convert_to_datetime(self, column_name, format=None):
        if format:
            self[column_name] = pd.to_datetime(self[column_name], format=format)
        else:
            self[column_name] = pd.to_datetime(self[column_name])

    def replace_column_str(self, column_name, value, new_val):
        self[column_name] = self[column_name].str.replace(value, new_val)

    def subset(self, length):
        df = self.ix[:length]
        df.__class__ = Table
        return df

    def mv_index_to_col(self, column_name=None):
        self.reset_index(inplace=True)
        if column_name:
            self.rename_column('index', column_name)

    def to_list_of_lists(self):
        data = self.values.tolist()
        if not isinstance(self.index, RangeIndex):
            new_data = [['__index__']]
            index_vals = self.index.tolist()
            for i, row in enumerate(data):
                new_data.append([index_vals[i]] + row)
            data = new_data
        return data

    def get_dict(self):
        column_names = list(self.columns)
        data = self.to_list_of_lists()
        if '__index__' in data[0]:
            column_names = ['__index__'] + column_names
        new_dict = {
            'columns': column_names,
            'data': data[1:],
        }
        return new_dict

    def print(self, underline=True):
        df_ll = self.to_list_of_lists()
        print(tabulate(df_ll, ['index']+list(self.columns), numalign="left", tablefmt="presto"))

    @staticmethod
    def format_ident(ident):
        if (isinstance(ident, int) or \
            isinstance(ident, str)):
            return ident
        elif isinstance(ident, int64):
            return int(ident)
        else:
            print('Type found: %s' % type(ident))
            raise NotImplementedError
        return ident

    @staticmethod
    def load(filepath, filetype=None, nrows=None, na_values=None):
        kwargs = {}
        if filetype == 'PLAINTEXT':
            kwargs['delimiter'] = "\n"
        if nrows:
            kwargs['nrows'] = nrows
        if na_values:
            kwargs['na_values'] = na_values
        df = pd.read_csv(filepath, **kwargs)
        df.__class__ = Table
        return df

    def copy(self):
        df = super(Table, self).copy()
        df.__class__ = Table
        return df

def get_random_df():
    df = DataFrame(np.random.rand(5,3), columns=('A','B','C'))
    df.__class__ = Table
    tbl = df
    return tbl

if __name__ == '__main__':
    print(get_random_df())