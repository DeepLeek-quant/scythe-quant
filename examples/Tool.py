import os
import numpy as np
import pandas as pd

import tqdm
import quantstats as qs
qs.extend_pandas()
import cufflinks as cf
cf.go_offline()
class Handler(dict):
    def __init__(self,path,data_type:str = 'parquet'):
        self.path = path
        self.cashe_dict = {}
        self.func_dict = {}
        if data_type == 'pickle':
            data_type = 'pkl'
        self.data_type = data_type
        self.reindex_like = None
        os.makedirs(path,exist_ok=True)
    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.path, f'{key}.{self.data_type}')
            # 检查存储的文件是否存在
            if os.path.exists(file_path):
                try:
                    if self.data_type == 'parquet':
                        if self.reindex_like is not None:
                            return pd.read_parquet(file_path).reindex_like(self.reindex_like)
                        else:
                            return pd.read_parquet(file_path)
                    elif self.data_type == 'pkl':
                        if self.reindex_like is not None:
                            return pd.read_pickle(file_path).reindex_like(self.reindex_like)
                        else:
                            return pd.read_pickle(file_path)
                except :
                    # 如果文件损坏或无法读取，返回默认值
                    raise ValueError(f'文件{file_path}损坏或无法读取')
            raise ValueError(f'文件{file_path}不存在')
    def __call__(self, key):
        return self.__getitem__(key)
    def __setitem__(self, key, value):
        file_path = os.path.join(self.path, f'{key}.{self.data_type}')
        value[np.isfinite(value)].to_parquet(file_path)
    def cash_list(self):
        parquet_set = set(filter(lambda X:X.endswith(f".{self.data_type}"),os.listdir(self.path)))
        return sorted(map(lambda X:X[:-(len(self.data_type)+1)],list(parquet_set)))

def max_drawdown(prices):
    # 計算累計的最大值
    cumulative_max = prices.cummax()
    # 計算回撤 (Drawdown)
    drawdown = (prices - cumulative_max) / cumulative_max
    # 計算最大回撤 (MDD)
    mdd = drawdown.min()
    return mdd
def show_stats(bt_ret:pd.DataFrame)->None:
    if isinstance(bt_ret,pd.Series):
        bt_ret = pd.DataFrame({'策略':bt_ret})
    try:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'Calmar':bt_ret.calmar(),
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('ME').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('ME').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('YE').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('YE').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
    except:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('M').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('M').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('Y').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('Y').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
def backtest_factor(factor:pd.DataFrame,exp_ret:pd.DataFrame,rank_range_n:int = 10,start_date:str = '2019-01-01'):
    factor_rank = factor.rank(axis = 1,pct = True,method = 'first')

    IC_Se = factor.corrwith(exp_ret,axis=1,method='spearman').sort_index().loc[start_date:]
    print(f'IC_mean:{round(IC_Se.mean(),4)}')
    print(f'IC_IR:{round(IC_Se.mean()/IC_Se.std(),4)}')

    bt = pd.concat({f'{int(((_/rank_range_n)*100))}% ~ {int((_+1)/rank_range_n*100)}%':exp_ret[(factor_rank>_/rank_range_n) & (factor_rank<=(_+1)/rank_range_n)].mean(axis = 1) - exp_ret.mean(axis=1) for _ in tqdm.tqdm(range(rank_range_n))}, axis = 1).dropna(how = 'all')
    bt = bt.loc[start_date:]
    if (bt.iloc[:,-1] - bt.iloc[:,0]).add(1).prod() > 1:
        bt['LS_ret'] = bt.iloc[:,-1] - bt.iloc[:,0]
    else:
        bt['LS_ret'] = bt.iloc[:,0] - bt.iloc[:,-1]
    show_stats(bt)

    (bt.drop(columns='LS_ret').loc[start_date:].cagr()*100).iplot(kind = 'bar')
    bt.index = bt.index.astype(str)
    bt.cumsum().ffill().iplot()
    bt.index = pd.to_datetime(bt.index)
CAGR = lambda bt_ret:(abs(bt_ret.add(1).prod())**(1/((bt_ret.index[-1] - bt_ret.index[0]).days / 365.)) - 1)*100