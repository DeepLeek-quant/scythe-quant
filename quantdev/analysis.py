from typing import Union, Literal, Tuple
import pandas as pd
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# utils


# factor analysis
def calc_factor_longshort_return(factor:Union[pd.DataFrame, str], asc:bool=True, rebalance:str='QR', group:int=10):
    from quantdev.backtest import get_factor, quick_backtesting
    factor = get_factor(item=factor, asc=asc)
    long = quick_backtesting(factor>=(1-1/group), rebalance=rebalance)
    short = quick_backtesting(factor<(1/group), rebalance=rebalance)

    return (long-short).dropna()

def calc_factor_quantiles_return(factor:pd.DataFrame, asc:bool=True, rebalance:str='QR', group:int=10):
    from quantdev.backtest import get_factor, quick_backtesting
    results = {}
    factor = get_factor(item=factor, asc=asc)
    for q_start, q_end in [(i/group, (i+1)/group) for i in range(group)]:
        condition = (q_start <= factor) & (factor < q_end)
        results[q_start*group] = quick_backtesting(condition, show=False, rebalance=rebalance)

    return results

def plot_factor(factor:pd.DataFrame, asc:bool=True, rebalance:str='QR', group:int=10):
    results = calc_factor_quantiles_return(factor, asc=asc, rebalance=rebalance, group=group)

    fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"colspan": 2, "secondary_y": True}, None],
                    [{}, {}],
                ],
                vertical_spacing=0.05,
                horizontal_spacing=0.1,
            )

    # relative return
    base_key = 0
    # for key in sorted(results.keys()):
    #     if (results[key]).isna().count() / len(pd.DataFrame(results)) >0.5:
    #         base_key = key
    #         break
    f_daily_rtn = (1 + results[max(results.keys())]).pct_change() - (1 + results[base_key]).pct_change()
    f_rtn  = (1+f_daily_rtn).cumprod()-1
    f_relative_rtn, downtrend_rtn = self._calc_relative_return(f_daily_rtn)
    bmk_rtn = self.bmk_equity_df

    # relative return
    strategy_trace = go.Scatter(
        x=f_relative_rtn.index,
        y=f_relative_rtn.values,
        name='Relative Return (lhs)',
        line=dict(color=self.fig_param['colors']['Dark Blue'], width=2),
        mode='lines',
        yaxis='y1',
    )

    # downtrend
    downtrend_trace = go.Scatter(
        x=f_relative_rtn.index,
        y=downtrend_rtn,
        name='Downtrend (lhs)',
        line=dict(color=self.fig_param['colors']['Bright Red'], width=2),
        mode='lines',
        yaxis='y1',
    )

    # factor return
    factor_return_trace = go.Scatter(
        x=f_rtn.index,
        y=f_rtn.values,
        name='Factor Return (rhs)',
        line=dict(color=self.fig_param['colors']['White'], width=2),
        mode='lines',
        yaxis='y2',
    )


    benchmark_trace = go.Scatter(
        x=bmk_rtn.index,
        y=bmk_rtn.values,
        name='Benchmark Return (rhs)',
        line=dict(color=self.fig_param['colors']['Dark Grey'], width=2),
        mode='lines',
        yaxis='y2',
    )

    fig.add_trace(strategy_trace, row=1, col=1)
    fig.add_trace(downtrend_trace, row=1, col=1)
    fig.add_trace(factor_return_trace, row=1, col=1, secondary_y=True)
    fig.add_trace(benchmark_trace, row=1, col=1, secondary_y=True)

    # c return
    num_lines = len(results)
    color_scale = sample_colorscale('PuBu', [n / num_lines for n in range(num_lines)])
    for (key, value), color in zip(reversed(list(results.items())), color_scale):
        annual_rtn = (1 + value.iloc[-1]) ** (240 / len(value)) - 1 if not value.empty else 0
        fig.add_trace(go.Scatter(
            x=value.index,
            y=value.values,
            name=int(key),
            line=dict(color=color),
            showlegend=False,
        ),
        row=2, col=1)

        # annual return
        fig.add_annotation(
            x=value.index[-1],
            y=value.values[-1], 
            xref="x", 
            yref="y", 
            text=f'{int(key)}: {annual_rtn: .2%}',  
            showarrow=False, 
            xanchor="left", 
            yanchor="middle", 
            font=dict(color="white", size=10),
        row=2, col=1)

    # IC
    from quantdev.backtest import Strategy
    ic = Strategy._calc_ic(factor)

    fig.add_trace(go.Scatter(
        x=ic.index, 
        y=ic.values, 
        mode='lines', 
        name='IC', 
        line=dict(color=self.fig_param['colors']['Bright Orange'], width=1),
        showlegend=False,
        ),
        row=2, col=2)
    fig.add_hline(
        y=0,
        line=dict(color="white", width=1),
        row=2, col=2
    )

    # rangeslider_visible
    fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=1, col=1)

    # position
    fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
    fig.update_xaxes(domain=[0.025, 0.45], row=2, col=1)
    fig.update_xaxes(domain=[0.55, 0.975], row=2, col=2)

    fig.update_yaxes(domain=[0.6, .95], row=1, col=1)
    fig.update_yaxes(domain=[0, 0.4], row=2, col=1)
    fig.update_yaxes(domain=[0, 0.4], row=2, col=2)

    # adjust axes
    fig.update_xaxes(tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False)
    fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=True, showgrid=False) # second y-axis
    fig.update_yaxes(tickformat=".0%", row=2, col=1)

    # add titles
    fig.add_annotation(text=self.bold(f'Factor Relative Returns'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=self.bold(f'Factor Return by Quantiles'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=self.bold(f'Information Coefficient(IC)'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

    ic_mean = ic.mean()
    ic_std = ic.std()
    info_ratio = ic_mean/ic_std
    positive_ratio = len(ic[ic>0])/len(ic)
    fig.add_annotation(text=f'Mean: {ic_mean :.2f}, SD: {ic_std :.2f}, IR: {info_ratio :.2f}, Positive Ratio: {positive_ratio:.0%}', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

    # wrap up
    fig.update_layout(
        legend=dict(x=0.05, y=0.94, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),    
        height=self.fig_param['size']['h'], 
        width=self.fig_param['size']['w'], 
        margin= self.fig_param['margin'],
        template=self.fig_param['template'],
    )
    return fig

# strategy analysis