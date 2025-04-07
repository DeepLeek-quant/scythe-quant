from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from numerize.numerize import numerize
import plotly.figure_factory as ff
import plotly.graph_objects as go
import panel as pn
import calendar

import pandas as pd
import numpy as np

_fig_param = dict(
    size=dict(w=800, h=600),
    margin=dict(t=50, b=50, l=50, r=50),
    template='plotly_dark',
    colors={
        'Bright Green':'#00FF00', 'Bright Red':'#FF0000', 'Bright Yellow':'#FFCC00', 
        'Bright Cyan':'#00FFFF', 'Bright Orange':'#FF9900', 'Bright Magenta':'#FF00FF',
        'Light Grey':'#AAAAAA', 'Light Blue':'#636EFA', 'Light Red':'#EF553B',
        'Dark Blue':'#0000FF', 'Dark Grey':'#7F7F7F', 'White':'#FFFFFF', 
    },
    font=dict(family='Arial'),
    pos_colorscale = [[0,'#414553'],[0.5,'#6E0001'],[1,'#FF0001']],
    neg_colorscale = [[0,'#02EF02'],[0.5,'#017301'],[1,'#405351']],
)

def _bold(text):
    return f'<b>{text}</b>'

def show_colors():
    fig = go.Figure()
    
    # Calculate positions
    colors = _fig_param['colors']
    y_start = 0.8
    spacing = 0.1
    x_base = 0.15
    
    for i, (name, color) in enumerate(colors.items()):
        y_pos = y_start - i*spacing
        
        # Lines
        fig.add_trace(go.Scatter(
            x=[x_base, x_base+0.1], y=[y_pos]*2,
            mode='lines', line=dict(color=color, width=4),
            name=f"{name} ({color})", showlegend=True
        ))
        # Dots
        fig.add_trace(go.Scatter(
            x=[x_base+0.2], y=[y_pos],
            mode='markers', marker=dict(color=color, size=12),
            showlegend=False
        ))
        # Boxes
        fig.add_shape(
            type="rect", fillcolor=color, line_color=color,
            x0=x_base+0.3, x1=x_base+0.4,
            y0=y_pos-0.02, y1=y_pos+0.02
        )
        # Text
        fig.add_trace(go.Scatter(
            x=[x_base+0.5], y=[y_pos],
            mode='text', text=f"{name} ({color})",
            textposition='middle right',
            textfont=dict(size=12, color='white'),
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            showlegend=False,
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False,
                    range=[y_start-len(colors)*spacing-0.1, y_start+0.1]),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False,
                    range=[0, 1]),
            height=_fig_param['size']['h'],
            width=_fig_param['size']['w'], 
            margin=_fig_param['margin'],
            template=_fig_param['template'],
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def plot_equity_curve(daily_returns:pd.DataFrame, benchmark_daily_returns:pd.DataFrame, portfolio_df:pd.DataFrame):
        
        c_return = (1 + daily_returns).cumprod() - 1
        bmk_c_return = (1 + benchmark_daily_returns).cumprod() - 1
        
        fig = make_subplots(
            rows=3, cols=1, 
            vertical_spacing=0.05, 
            shared_xaxes=True,
            )

        # equity curve
        fig.append_trace(go.Scatter(
            x=bmk_c_return.index,
            y=bmk_c_return.values,
            showlegend=False,
            name='Benchmark',
            line=dict(color=_fig_param['colors']['Dark Grey'], width=2),
            ), row=1, col=1)
        fig.append_trace(go.Scatter(
            x=c_return.index,
            y=c_return.values,
            name='Strategy',
            line=dict(color=_fig_param['colors']['Dark Blue'], width=2),
            ), row=1, col=1)

        # drawdown
        drawdown = (c_return+1 - (c_return+1).cummax()) / (c_return+1).cummax()
        bmk_drawdown = (bmk_c_return+1 - (bmk_c_return+1).cummax()) / (bmk_c_return+1).cummax()

        fig.append_trace(go.Scatter(
            x=bmk_drawdown.index,
            y=bmk_drawdown.values,
            name='Benchmark-DD',
            showlegend=False,
            line=dict(color=_fig_param['colors']['Dark Grey'], width=2),
            ), row=2, col=1)
        fig.append_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Strategy-DD',
            line=dict(color=_fig_param['colors']['Bright Magenta'], width=1.5),
            ), row=2, col=1)

        # portfolio size
        p_size = portfolio_df.apply(lambda row: np.count_nonzero(row), axis=1)
        fig.append_trace(go.Scatter(
            x=p_size.index,
            y=p_size.values,
            name='Strategy-size',
            line=dict(color=_fig_param['colors']['Bright Orange'], width=2),
            ), row=3, col=1)

        # adjust time range
        fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=3, col=1)

        # asjust y as % for equity curve & drawdown
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)

        # position
        fig.update_yaxes(domain=[0.5, 0.95], row=1, col=1)
        fig.update_yaxes(domain=[0.2, 0.4], row=2, col=1)
        fig.update_yaxes(domain=[0, 0.1], row=3, col=1)

        # titles
        fig.add_annotation(text=_bold(f'Cumulative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=_bold(f'Drawdown'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=_bold(f'Portfolio Size'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

        # fig layout
        fig.update_layout(
            legend=dict(x=0.01, y=0.95, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
            width=_fig_param['size']['w'],
            height=_fig_param['size']['h'],
            margin=_fig_param['margin'],
            template=_fig_param['template'],
        )
        return fig

def plot_return_heatmap(daily_returns:pd.DataFrame):
    # prepare data
    monthly_return = ((1 + daily_returns).resample('ME').prod() - 1)\
        .reset_index()\
        .assign(
            y=lambda x: x['t_date'].dt.year.astype('str'),
            m=lambda x: x['t_date'].dt.month,
            )
    annual_return = (daily_returns + 1).groupby(daily_returns.index.year).prod() - 1
    my_heatmap = monthly_return.pivot(index='y', columns='m', values=0).iloc[::-1]
    avg_mth_return = monthly_return.groupby('m')[0].mean()

    # plot
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1)
    pos_colorscale = _fig_param['pos_colorscale']
    neg_colorscale = _fig_param['neg_colorscale']

    # annual return heatmap
    annual_return_pos = np.where(annual_return > 0, annual_return, np.nan)
    annual_return_neg = np.where(annual_return <= 0, annual_return, np.nan)

    # annual return for positive values
    fig.append_trace(
        go.Heatmap(
            z=annual_return_pos.reshape(1, -1),
            x=annual_return.index.astype('str'),
            y=[''], 
            text=annual_return.apply(lambda x: f"{x:.1%}").values.reshape(1, -1),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorscale = pos_colorscale,
            showscale = False,
            zmin=0,
            zmax=0.4,
        ),
        row=1, col=1
    )

    # annual return for negative values
    fig.append_trace(
        go.Heatmap(
            z=annual_return_neg.reshape(1, -1),
            x=annual_return.index.astype('str'),
            y=[''], 
            text=annual_return.apply(lambda x: f"{x:.1%}").values.reshape(1, -1),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorscale = neg_colorscale,
            showscale = False,
            zmin=-0.4,
            zmax=0,
        ),
        row=1, col=1
    )


    # monthly return heatmap
    my_heatmap_pos = np.where(my_heatmap > 0, my_heatmap, np.nan)
    my_heatmap_neg = np.where(my_heatmap <= 0, my_heatmap, np.nan)

    # monthly return for positive values
    fig.append_trace(
        go.Heatmap(
            z=my_heatmap_pos,
            x=my_heatmap.columns,
            y=my_heatmap.index,
            text=np.where(my_heatmap_pos > 0, np.vectorize(lambda x: f"{x:.1%}")(my_heatmap_pos), ""),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale = pos_colorscale,
            showscale = False,
            zmin=0,
            zmax=0.1,
        ),
        row=2, col=1
    )

    # monthly return for negative values
    fig.append_trace(
        go.Heatmap(
            z=my_heatmap_neg,
            x=my_heatmap.columns,
            y=my_heatmap.index,
            text=np.where(my_heatmap_neg <= 0, np.vectorize(lambda x: f"{x:.1%}")(my_heatmap_neg), ""),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale = neg_colorscale,
            showscale = False,
            zmin=-0.1,
            zmax=0,
        ),
        row=2, col=1
    )

    # monthly average return
    fig.append_trace(
    go.Bar(
        x=avg_mth_return.index,
        y=avg_mth_return.values,
        marker=dict(color=self.fig_param['colors']['Bright Orange']),
        name='Avg. return',
        
    ),
    row=3, col=1
    )
    fig.add_hline(
        y=0,
        line=dict(color="white", width=1),
        row=3, col=1
    )

    # adjust for axes display
    fig.update_xaxes(tickfont=dict(size=11), row=1, col=1)
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[calendar.month_abbr[i] for i in range(1, 13)], row=2, col=1)
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=[calendar.month_abbr[i] for i in range(1, 13)], row=3, col=1, matches='x2')
    fig.update_yaxes(tickformat=".01%", row=3, col=1)

    # position
    fig.update_yaxes(domain=[0.9, 0.95], row=1, col=1)
    fig.update_yaxes(domain=[0.3, 0.75], row=2, col=1)
    fig.update_yaxes(domain=[0, 0.15], row=3, col=1)

    # titles
    fig.add_annotation(text=_bold('Annual Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold('Monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=_bold('Avg. monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

    # fig size, colorscale & bgcolor
    fig.update_layout(
        width=self.fig_param['size']['w'],
        height=self.fig_param['size']['h'],
        margin=self.fig_param['margin'],
        template=self.fig_param['template'],
    )

    return fig

def plot_liquidity(portfolio_df:pd.DataFrame=None, money_threshold:int=500000, volume_threshold:int=50):
        df=self._liquidity_analysis(portfolio_df)

        # lqd_heatmap
        low_money = (df['成交金額(元)'] < money_threshold).mean()
        low_volume = (df['成交量(千股)'] < volume_threshold).mean()
        全額交割 = (df['是否全額交割']=='Y').mean()
        注意股票 = (df['是否為注意股票']=='Y').mean()
        處置股票 = (df['是否為處置股票']=='Y').mean()
        buy_limit = ((df['date']==df['sell_date'])&(df['漲跌停註記'] == '-')).mean()
        sell_limit = ((df['date']==df['buy_date'])&(df['漲跌停註記'] == '+')).mean()
        lqd_heatmap = pd.DataFrame({      
            f'money < {numerize(money_threshold)}': [low_money],
            f'volume < {numerize(volume_threshold)}': [low_volume],
            '全額交割': [全額交割],
            '注意股票': [注意股票],
            '處置股票': [處置股票],
            'buy limit up': [buy_limit],
            'sell limit down': [sell_limit],
            },
            index=[0],
        )

        # safety capacity
        principles = [500000, 1000000, 5000000, 10000000, 50000000, 100000000]
        buys_lqd = df[df['buy_date']==df['date']]
        vol_ratio = 10
        capacity = pd.DataFrame({
            'principle': [numerize(p) for p in principles], # Format principle for better readability
            'safety_capacity': [
                (buys_lqd.assign(
                capacity=lambda x: x['成交金額(元)'] / vol_ratio - (principle / x.groupby('buy_date')['stock_id'].transform('nunique')),
                spill=lambda x: np.where(x['capacity'] < 0, x['capacity'], 0)
                )
                .groupby('buy_date')['spill'].sum() * -1 / principle).mean()
                for principle in principles
                ]
        })
        capacity['safety_capacity'] = 1 - capacity['safety_capacity']

        # entry, exit volume
        def volume_threshold(df):
            thresholds = [100000, 500000, 1000000, 10000000, 100000000]
            labels = [f'{numerize(thresholds[i-1])}-{numerize(th)}' 
                        if i > 0 else f'<= {numerize(th)}'
                        for i, th in enumerate(thresholds)]
            labels.append(f'> {numerize(thresholds[-1])}')
            
            return df.assign(
                money_threshold=pd.cut(df['成交金額(元)'], [0] + thresholds + [np.inf], labels=labels)
                )\
                .groupby('money_threshold', observed=True)['stock_id']\
                .count()\
                .pipe(lambda x: (x / x.sum()))

        ee_volume = pd.concat([
            volume_threshold(df[df['buy_date']==df['date']]).rename('buy'),
            volume_threshold(df[df['sell_date']==df['date']]).rename('sell')
        ], axis=1)

        # plot
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{}, {},]
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            )

        # lqd_heatmap
        fig.add_trace(go.Heatmap(
            z=lqd_heatmap.loc[0].values.reshape(-1, 1), 
            x=lqd_heatmap.columns.values, 
            y=[''], 
            text = lqd_heatmap.loc[0].apply(lambda x: f"{x:.2%}").values.reshape(1, -1),
            texttemplate="%{text}",
            transpose=True,
            textfont=dict(size=10),
            colorscale = 'Plotly3',
            showscale = False,
            zmin=0,
            zmax=0.1,
        ),
        row=1, col=1)

        # safety capacity
        fig.add_trace(go.Bar(
            x=capacity['principle'],
            y=capacity['safety_capacity'],
            showlegend=False,
            marker_color=_fig_param['colors']['Bright Orange'],
            name='Safety cap',
            text=capacity.safety_capacity.apply(lambda x: f"{x:.2%}").values,
            textposition='inside',
            width=0.7
        ),
        row=2, col=1)

        # entry, exit volume
        fig.add_trace(go.Bar(
            x=ee_volume.index,
            y=ee_volume.buy.values,
            name='Entry',
            marker_color=_fig_param['colors']['Bright Cyan'], 
        ),
        row=2, col=2)
        fig.add_trace(go.Bar(
            x=ee_volume.index,
            y=ee_volume.sell.values,
            name='Exit',
            marker_color=_fig_param['colors']['Bright Red'], 
        ),
        row=2, col=2)

        # adjust axes
        fig.update_xaxes(tickfont=dict(size=11), side='top', row=1, col=1)
        fig.update_xaxes(tickfont=dict(size=11), title_text='total capital', row=2, col=1)
        fig.update_xaxes(tickfont=dict(size=11), title_text="trading money", row=2, col=2)

        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=2)

        # add titles
        fig.add_annotation(text=_bold('Liquidity Heatmap'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
        fig.add_annotation(text=_bold('Safety Capacity'), align='left', x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
        fig.add_annotation(text=f'(Ratio of buy money ≤ 1/{vol_ratio} of volume for all trades)', align='left', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=1)
        fig.add_annotation(text=_bold(f'Trading Money at Entry/Exit'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

        # position
        fig.update_yaxes(domain=[0.87, 0.92], row=1, col=1)
        fig.update_yaxes(domain=[0.2, 0.65], row=2, col=1, range=[0, 1])
        fig.update_yaxes(domain=[0.2, 0.65], row=2, col=2, dtick=0.1)

        # laoyout
        fig.update_layout(
            legend=dict(x=0.55, y=0.65, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
            coloraxis1_showscale=False,
            coloraxis1=dict(colorscale='Plasma'),
            width=_fig_param['size']['w'],
            height=_fig_param['size']['h'],
            margin=_fig_param['margin'],
            template=_fig_param['template'],
            )
        return fig