from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from numerize.numerize import numerize
import plotly.figure_factory as ff
import plotly.graph_objects as go
import calendar

from typing import Union, Tuple
import pandas as pd
import numpy as np

fig_param = {
    'size': {'w': 800, 'h': 600},
    'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50},
    'template': 'plotly_dark',
    'colors': {
        'Bright Green': '#00FF00', 'Bright Red': '#FF0000', 'Bright Yellow': '#FFCC00', 
        'Bright Cyan': '#00FFFF', 'Bright Orange': '#FF9900', 'Bright Magenta': '#FF00FF',
        'Light Grey': '#AAAAAA', 'Light Blue': '#636EFA', 'Light Red': '#EF553B',
        'Dark Blue': '#0000FF', 'Dark Grey': '#7F7F7F', 'White': '#FFFFFF', 
    },
    'font': {'family': 'Arial'},
    'pos_colorscale': [[0, '#414553'], [0.5, '#6E0001'], [1, '#FF0001']],
    'neg_colorscale': [[0, '#02EF02'], [0.5, '#017301'], [1, '#405351']],
}

# utils
def _bold(text):
    return f'<b>{text}</b>'

def show_colors():
    fig = go.Figure()
    
    # Calculate positions
    colors = fig_param['colors']
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
        height=fig_param['size']['h'],
        width=fig_param['size']['w'], 
        margin=fig_param['margin'],
        template=fig_param['template'],
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# plot
def plot_equity_curve(equity_curve:Union[pd.DataFrame, pd.Series], benchmark_equity_curve:Union[pd.DataFrame, pd.Series]=None, portfolio_size:Union[pd.DataFrame, pd.Series]=None)-> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.05, 
        shared_xaxes=True,
        )

    # benchmark
    benchmark_equity_curve = benchmark_equity_curve.reindex_like(equity_curve)
    benchmark_dd = (benchmark_equity_curve+1 - (benchmark_equity_curve+1).cummax()) / (benchmark_equity_curve+1).cummax()
    fig.append_trace(go.Scatter(
        x=benchmark_equity_curve.index,
        y=benchmark_equity_curve.values,
        showlegend=False,
        name='Benchmark',
        line=dict(color=fig_param['colors']['Dark Grey'], width=1.5),
        ), row=1, col=1)
    fig.append_trace(go.Scatter(
        x=benchmark_dd.index,
        y=benchmark_dd.values,
        name='Benchmark-DD',
        showlegend=False,
        line=dict(color=fig_param['colors']['Dark Grey'], width=1.5),
        ), row=2, col=1)
    
    # strategy
    equity_curve = pd.DataFrame(equity_curve.rename('Strategy')) if isinstance(equity_curve, pd.Series) else equity_curve
    dd = (equity_curve+1 - (equity_curve+1).cummax()) / (equity_curve+1).cummax()

    # Generate color scale for multiple strategies
    n_strategies = len(equity_curve.columns)
    color_scale = {}
    if n_strategies > 1:
        colors = sample_colorscale('Teal', [0.6 + (n * 0.4/(n_strategies-1)) for n in range(n_strategies)])
        color_scale = dict(zip(equity_curve.columns, colors))

    # Plot equity curves and drawdowns for each strategy
    for col in equity_curve.columns:
        # Add equity curve
        fig.append_trace(go.Scatter(
            x=equity_curve.index, 
            y=equity_curve[col].values, 
            name=col,
            legendgroup=col, 
            legendrank=10000+equity_curve.columns.get_loc(col),
            line=dict(color=color_scale.get(col, fig_param['colors']['Dark Blue']), width=1.5),
        ), row=1, col=1)
        
        # Add drawdown curve
        fig.append_trace(go.Scatter(
            x=dd.index, 
            y=dd[col].values, 
            name=f'{col}-DD',
            legendgroup=col, 
            legendrank=10000+dd.columns.get_loc(col),
            showlegend=False,
            line=dict(color=color_scale.get(col, fig_param['colors']['Bright Magenta']), width=1.5),
        ), row=2, col=1)
    
    # portfolio size
    portfolio_size = pd.Series('', index=equity_curve.index) if portfolio_size is None else portfolio_size
    fig.append_trace(go.Scatter(
        x=portfolio_size.index,
        y=portfolio_size.values,
        name='Strategy-size',
        showlegend=False,
        line=dict(color=fig_param['colors']['Bright Orange'], width=1.5),
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
        width=fig_param['size']['w'],
        height=fig_param['size']['h'],
        margin=fig_param['margin'],
        template=fig_param['template'],
    )
    return fig

def plot_return_heatmap(daily_return:Union[pd.DataFrame, pd.Series])-> go.Figure:
    # prepare data
    if isinstance(daily_return, pd.DataFrame):
        daily_return = daily_return.iloc[:, 0]
    daily_return = daily_return.rename(0)
    monthly_return = ((1 + daily_return).resample('ME').prod() - 1)\
        .reset_index()\
        .assign(
            y=lambda x: x['t_date'].dt.year.astype('str'),
            m=lambda x: x['t_date'].dt.month,
            )
    annual_return = (daily_return + 1).groupby(daily_return.index.year).prod() - 1
    my_heatmap = monthly_return.pivot(index='y', columns='m', values=0).iloc[::-1]
    avg_mth_return = monthly_return.groupby('m')[0].mean()

    # plot
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1)
    pos_colorscale = fig_param['pos_colorscale']
    neg_colorscale = fig_param['neg_colorscale']

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
            marker=dict(color=fig_param['colors']['Bright Orange']),
            name='Avg. return', 
        ), 
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color="white", width=1), row=3, col=1)

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
    fig.add_annotation(text=_bold(f'Annual Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=_bold(f'Avg. monthly Return'), x=0, y=1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=3, col=1)

    # fig size, colorscale & bgcolor
    fig.update_layout(
        width=fig_param['size']['w'],
        height=fig_param['size']['h'],
        margin=fig_param['margin'],
        template=fig_param['template'],
    )

    return fig

def plot_style(betas:pd.DataFrame)-> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        specs=[
            [{"secondary_y": True}],
            [{}],
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
    )

    # rolling beta
    alpha = (1 + betas.iloc[:-1]['const']).cumprod() - 1
    betas = betas.drop(columns=['const'])
    betas = betas.reindex(columns=betas.iloc[-1].sort_values(ascending=False).index).round(3)

    colors = sample_colorscale('Spectral', len(betas.columns))
    for i, col in enumerate(betas.columns):
        rolling_beta = betas.iloc[:-1]
        total_beta = betas.iloc[-1]
        # return (total_beta)
        fig.add_trace(
            go.Scatter(
                x=rolling_beta.index, 
                y=rolling_beta[col], 
                name=col,
                line=dict(color=colors[i]),
                showlegend=True,
                legendgroup=col,
            ),
            secondary_y=True,
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[col],
                y=[total_beta[col]],
                text=[f'{total_beta[col]:.3f}'],
                textposition='auto',
                marker_color=colors[i],
                name=col,
                showlegend=False,
                legendgroup=col,
                width=0.6,
            ),
            row=2, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=alpha.index, 
            y=alpha.values, 
            name='Alpha (rhs)',
            line=dict(color=fig_param['colors']['Light Grey'], width=2),
            showlegend=True,
        ),
        secondary_y=False,
        row=1, col=1,
    )

    # adjust axes
    fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(row=2, col=1, showgrid=False)

    # position
    fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
    fig.update_xaxes(domain=[0.025, 0.975], row=2, col=1)

    fig.update_yaxes(domain=[0.55, 0.975], row=1, col=1)
    fig.update_yaxes(domain=[0, 0.3], row=2, col=1)

    # titles
    fig.add_annotation(text=_bold('Rolling beta'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold('Total beta'), x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

    # layout
    fig.update_layout(
        legend = dict(x=0.05, y=0.5, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', tracegroupgap=.25, orientation='h'),
        width = fig_param['size']['w'],
        height = fig_param['size']['h'],
        margin = fig_param['margin'],
        template = fig_param['template'],
        yaxis = dict(side='right'),
        yaxis2 = dict(side='left'),
    )
    fig.update_polars(radialaxis_showline=False)

    return fig

def plot_liquidity(lq_df:pd.DataFrame, volume_threshold:int=50)-> go.Figure:

    # lqd_heatmap
    full_settle = (lq_df['是否全額交割']=='Y')
    notice = (lq_df['是否為注意股票']=='Y')
    disposal = (lq_df['是否為處置股票']=='Y')
    buy_limit = ((lq_df['date']==lq_df['sell_date'])&(lq_df['漲跌停註記'] == '-'))
    sell_limit = ((lq_df['date']==lq_df['buy_date'])&(lq_df['漲跌停註記'] == '+'))
    low_volume = (lq_df['成交量']<50)
    low_money = (lq_df['成交金額']<5e5)
    lqd_timeline = pd.concat([
        lq_df[low_money].groupby('date')['stock_id'].count().rename(f'成交量<50萬: {low_money.mean():.2%}'),
        lq_df[low_volume].groupby('date')['stock_id'].count().rename(f'成交量<50張: {low_volume.mean():.2%}'),
        lq_df[full_settle].groupby('date')['stock_id'].count().rename(f'全額交割股: {full_settle.mean():.2%}'),
        lq_df[notice].groupby('date')['stock_id'].count().rename(f'注意股: {notice.mean():.2%}'),
        lq_df[disposal].groupby('date')['stock_id'].count().rename(f'處置股: {disposal.mean():.2%}'),
        lq_df[buy_limit].groupby('date')['stock_id'].count().rename(f'買在漲停: {buy_limit.mean():.2%}'),
        lq_df[sell_limit].groupby('date')['stock_id'].count().rename(f'賣在跌停: {sell_limit.mean():.2%}'),
    ], axis=1).fillna(0)

    # safety capacity
    principles = [500000, 1000000, 5000000, 10000000, 50000000, 100000000]
    buys_lqd = lq_df[lq_df['buy_date']==lq_df['date']]
    vol_ratio = 10
    capacity = pd.DataFrame({
        'principle': [numerize(p) for p in principles], # Format principle for better readability
        'safety_capacity': [
            (buys_lqd.assign(
            capacity=lambda x: x['成交金額'] / vol_ratio - (principle / x.groupby('buy_date')['stock_id'].transform('nunique')),
            spill=lambda x: np.where(x['capacity'] < 0, x['capacity'], 0)
            )
            .groupby('buy_date')['spill'].sum() * -1 / principle).mean()
            for principle in principles
            ]
    })
    capacity['safety_capacity'] = 1 - capacity['safety_capacity']

    # entry, exit volume
    def volume_threshold(data):
        thresholds = [100000, 500000, 1000000, 10000000, 100000000]
        labels = [f'{numerize(thresholds[i-1])}-{numerize(th)}' 
                    if i > 0 else f'<= {numerize(th)}'
                    for i, th in enumerate(thresholds)]
        labels.append(f'> {numerize(thresholds[-1])}')
        
        return data\
            .assign(money_threshold=pd.cut(data['成交金額'], [0, *thresholds, np.inf], labels=labels))\
            .groupby('money_threshold', observed=True)['stock_id']\
            .count()\
            .pipe(lambda x: (x / x.sum()))

    ee_volume = pd.concat([
        volume_threshold(lq_df[lq_df['buy_date']==lq_df['date']]).rename('buy'),
        volume_threshold(lq_df[lq_df['sell_date']==lq_df['date']]).rename('sell')
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

    # lqd timeline
    colors = sample_colorscale('Plotly3', [n/(lqd_timeline.shape[1]) for n in range(lqd_timeline.shape[1])])
    color_scale = dict(zip(lqd_timeline.columns, colors))

    for col in lqd_timeline.columns:
        fig.add_trace(go.Scatter(
            x=lqd_timeline.index,
            y=lqd_timeline[col],
            mode='lines',
            # line=dict(width=1), 
            line=dict(color=color_scale.get(col), width=1.5),
            name=col,
        ),
        row=1, col=1)

    # safety capacity
    fig.add_trace(go.Bar(
        x=capacity['principle'],
        y=capacity['safety_capacity'],
        showlegend=False,
        marker_color=fig_param['colors']['Bright Orange'],
        name='Safety cap',
        text=capacity['safety_capacity'].apply(lambda x: f"{x:.2%}").values,
        textposition='inside',
        width=0.7
    ),
    row=2, col=1)

    # entry, exit volume
    fig.add_trace(go.Bar(
        x=ee_volume.index,
        y=ee_volume.buy.values,
        name='Entry',
        marker_color=fig_param['colors']['Bright Cyan'], 
        showlegend=False,
    ),
    row=2, col=2)
    fig.add_trace(go.Bar(
        x=ee_volume.index,
        y=ee_volume.sell.values,
        name='Exit',
        marker_color=fig_param['colors']['Bright Red'], 
        showlegend=False,
    ),
    row=2, col=2)

    # adjust axes
    fig.update_xaxes(tickfont=dict(size=11), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=11), title_text='total capital', title_font=dict(size=12), row=2, col=1)
    fig.update_xaxes(tickfont=dict(size=11), title_text="trading money", title_font=dict(size=12), row=2, col=2)

    fig.update_yaxes(tickformat=".0f", title_text='size', title_font=dict(size=12), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=2)

    # add titles
    fig.add_annotation(text=_bold(f'Liquidity Heatmap'), x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Safety Capacity'), align='left', x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=f'(Ratio of buy money ≤ 1/{vol_ratio} of volume for all trades)', align='left', x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=1)
    fig.add_annotation(text=_bold(f'Trading Money at Entry/Exit'), x=0, y = 1, yshift=50, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)

    # position
    fig.update_yaxes(domain=[0.6, 0.97], row=1, col=1)
    fig.update_yaxes(domain=[0.1, 0.4], row=2, col=1, range=[0, 1])
    fig.update_yaxes(domain=[0.1, 0.4], row=2, col=2, dtick=0.1)

    # laoyout
    fig.update_layout(
        legend=dict(x=0, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', groupclick='togglegroup', itemclick='toggleothers'),
        coloraxis1_showscale=False,
        coloraxis1=dict(colorscale='Plasma'),
        width=fig_param['size']['w'],
        height=fig_param['size']['h'],
        margin=fig_param['margin'],
        template=fig_param['template'],
        )
    return fig

def plot_maemfe(maemfe:pd.DataFrame)-> go.Figure:
    win = maemfe[maemfe['return']>0]
    lose = maemfe[maemfe['return']<=0]

    fig = make_subplots(
        rows=2, cols=3,
        vertical_spacing=0,
        horizontal_spacing=0,
        )

    # colors
    light_blue = fig_param['colors']['Light Blue']
    light_red = fig_param['colors']['Light Red']
    dark_blue = fig_param['colors']['Dark Blue']
    dark_red = fig_param['colors']['Bright Red']

    # return distribution
    fig.add_trace(go.Histogram(
        x=win['return'], 
        marker_color=light_blue, 
        name='win',
        legendgroup='win',
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=lose['return'], 
        marker_color=light_red, 
        name ='lose',
        legendgroup ='lose',
    ), row=1, col=1)
    fig.add_vline(
        x=maemfe['return'].mean(), 
        line_width=2, 
        line_dash="dash", 
        line_color="Blue",
        annotation_text=f'avg. rtn: {maemfe["return"].mean() :.2%}',
        annotation_font_color='Blue',
    row=1, col=1)
    
    # bmfe/ mae
    fig.add_trace(go.Scatter(
        x = win['gmfe'], 
        y = win['mae']*-1, 
        mode='markers', 
        marker_color=light_blue, 
        text=win.index,
        textposition="top center",
        name='win',
        legendgroup='win',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x = lose['gmfe'], 
        y = lose['mae']*-1, 
        mode='markers', 
        marker_color=light_red, 
        text=lose.index,
        textposition="top center",
        name='lose',
        legendgroup ='lose',
    ), row=1, col=2)
    
    # gmfe/mdd
    fig.add_trace(go.Scatter(
        x = win['gmfe'], 
        y = win['mdd']*-1, 
        mode='markers', 
        marker_color=light_blue, 
        name='win',
        text=win.index,
        textposition="top center",
        legendgroup='win',
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x = lose['gmfe'], 
        y = lose['mdd']*-1, 
        mode='markers', 
        marker_color=light_red, 
        name='lose',
        text=lose.index,
        textposition="top center",
        legendgroup ='lose',
    ), row=1, col=3)
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        xref='x3', yref='y3',
        line=dict(color="Orange", width=2, dash="dash")
    )

    # positions
    fig.update_xaxes(tickformat=".0%", domain=[0, 0.25], title_text='return', row=1, col=1)
    fig.update_xaxes(tickformat=".0%", domain=[0.25+0.125, 0.5+0.125], title_text="gmfe", row=1, col=2)
    fig.update_xaxes(tickformat=".0%", domain=[0.75, 1], title_text="gmfe", row=1, col=3)
    
    fig.update_yaxes(domain=[0.625, 0.95], title_text='count', title_standoff=2, row=1, col=1)
    fig.update_yaxes(tickformat=".0%", domain=[0.625, 0.95], title_text='mae', title_standoff=2, row=1, col=2)
    fig.update_yaxes(tickformat=".0%", domain=[0.625, 0.95], title_text='mdd', title_standoff=2, row=1, col=3)

    # distributions
    def plot_distributions(item, x):
        jitter = 1e-10
        win_data = x*win[item] + jitter
        lose_data = x*lose[item] + jitter
        
        distplot=ff.create_distplot(
            [win_data, lose_data], 
            ['win', 'lose'], 
            colors=['#636EFA', '#EF553B'],
            bin_size=0.01, 
            histnorm='probability', 
            # curve_type='normal',
            show_rug=False
        )
        distplot.update_traces(dict(marker_line_width=0))
        return distplot

    distributions={
        'mae':{'row':2, 'col':1},
        'bmfe':{'row':2, 'col':2},
        'gmfe':{'row':2, 'col':3},
    }

    for key, value in distributions.items():
        x = -1 if key =='mae' else 1
        distplot = plot_distributions(key, x)
        
        # histogram
        fig.add_trace(go.Histogram(
            distplot['data'][0], 
            name='win',
            legendgroup='win', 
        ), row=value.get('row'), col=value.get('col'))
        fig.add_trace(go.Histogram(
            distplot['data'][1], 
            legendgroup ='lose', 
        ), row=value.get('row'), col=value.get('col'))
        
        # curve lines
        fig.add_trace(go.Scatter(
            distplot['data'][2],
            marker_color='blue'
        ), row=value.get('row'), col=value.get('col'))
        fig.add_trace(go.Scatter(
            distplot['data'][3],
            marker_color='red'
        ), row=value.get('row'), col=value.get('col'))
        
        # add v lines & annotations
        q3_win, q3_lose  = (x*win[key]).quantile(0.75), (x*lose[key]).quantile(0.75)
        fig.add_vline(
            x=q3_win, 
            line_width=2, line_dash="dash", line_color=dark_blue,
            row=value.get('row'), col=value.get('col'))
        fig.add_vline(
            x=q3_lose, 
            line_width=2, line_dash="dash", line_color=dark_red,
            row=value.get('row'), col=value.get('col'))
        fig.add_annotation(
            x=q3_win,
            y = 1,
            yref="y domain",
            # y='paper',
            text=f'Q3 {key} win: {q3_win:.2%}',
            showarrow=False,
            xshift=70,
            font=dict(color=dark_blue),
            row=value.get('row'), col=value.get('col'))
        fig.add_annotation(
            x=q3_lose,
            y = 0.9,
            yref="y domain",
            text=f'Q3 {key} lose: {q3_lose:.2%}',
            showarrow=False,
            xshift=70,
            # yshift=-20,
            font=dict(color=dark_red),
            row=value.get('row'), col=value.get('col'))
        
        # update yaxes
        fig.update_yaxes(domain=[0.05, 0.425], row=value.get('row'), col=value.get('col'))

        # update xaxes
        col=value.get('col')
        domain_start = (col - 1) * 0.375
        domain_end = domain_start + 0.25
        fig.update_xaxes(tickformat=".0%", domain=[domain_start, domain_end], title_text=key, row=value.get('row'), col=value.get('col'))

    # add titles
    fig.add_annotation(text=_bold(f'Return Distribution<br>win rate: {len(win)/len(maemfe):.2%}'), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'GMFE/ MAE'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=2)
    fig.add_annotation(text=_bold(f"GMFE/ MDD<br>missed profit pct:{len(win[win['mdd']*-1 > win['gmfe']])/len(win):.2%}"), x=0.5, y = 1, yshift=45, xref="x domain", yref="y domain", showarrow=False, row=1, col=3)
    fig.add_annotation(text=_bold(f'MAE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=_bold(f'BMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
    fig.add_annotation(text=_bold(f'GMFE distribution'), x=0.5, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=2, col=3)

    # layout
    fig.update_layout(
        width=fig_param['size']['w'],
        height=fig_param['size']['h'],
        margin=fig_param['margin'],
        template=fig_param['template'],
    )
    
    return fig

def plot_relative_return(relative_return:pd.DataFrame)-> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": True}, None],
            [{}, {}],
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        shared_xaxes=True,
    )

    # relative return
    strategy_trace = go.Scatter(
        x=relative_return.index,
        y=relative_return['relative_return'].values,
        name='Relative Return (lhs)',
        line=dict(color=fig_param['colors']['Dark Blue'], width=2),
        mode='lines',
        yaxis='y1'
    )

    downtrend_trace = go.Scatter(
        x=relative_return.index,
        y=relative_return['downtrend_return'].values,
        name='Downtrend (lhs)',
        line=dict(color=fig_param['colors']['Bright Red'], width=2),
        mode='lines',
        yaxis='y1'
    )

    benchmark_trace = go.Scatter(
        x=relative_return.index,
        y=relative_return['benchmark_return'].values,
        name='Benchmark Return (rhs)',
        line=dict(color=fig_param['colors']['Dark Grey'], width=2),
        mode='lines',
        yaxis='y2'
    )

    fig.add_trace(strategy_trace, row=1, col=1)
    fig.add_trace(downtrend_trace, row=1, col=1)
    fig.add_trace(benchmark_trace, row=1, col=1, secondary_y=True)  

    # rolling beta
    beta_trace = go.Scatter(
        x=relative_return.index,
        y=relative_return['beta'].values,
        name='Rolling Beta',
        line=dict(color=fig_param['colors']['Bright Orange'], width=1),
        showlegend=False,
    )
    
    fig.add_trace(beta_trace, row=2, col=1)
    fig.add_hline(
        y=relative_return['beta'].mean(),
        line=dict(color="white", width=1),
        line_dash="dash",
        annotation_text=f'avg. beta: {relative_return["beta"].mean() :.2}',
        annotation_position="bottom left",
        annotation_textangle = 0,
        row=2, col=1
    )

    # information ratio
    info_ratio_trace = go.Scatter(
        x=relative_return.index,
        y=relative_return['info_ratio'].values,
        name='Rolling info Ratio',
        line=dict(color=fig_param['colors']['Bright Magenta'], width=1),
        showlegend=False,
    )
    fig.add_trace(info_ratio_trace, row=2, col=2)
    fig.add_hline(
        y=relative_return['info_ratio'].mean(),
        line=dict(color="white", width=1),
        line_dash="dash",
        annotation_text=f'avg. info ratio: {relative_return["info_ratio"].mean() :.2}',
        annotation_position="bottom left",
        annotation_textangle = 0,
        row=2, col=2
    )

    # rangeslider_visible
    fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(thickness=0.01), row=1, col=1)

    # adjust axes
    fig.update_xaxes(tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=False, dtick=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1, secondary_y=True, showgrid=False) # second y-axis

    # position
    fig.update_xaxes(domain=[0.025, 0.975], row=1, col=1)
    fig.update_xaxes(domain=[0.025, 0.45], row=2, col=1)
    fig.update_xaxes(domain=[0.55, 0.975], row=2, col=2)

    fig.update_yaxes(domain=[0.6, .95], row=1, col=1)
    fig.update_yaxes(domain=[0, 0.4], row=2, col=1)
    fig.update_yaxes(domain=[0, 0.4], row=2, col=2)

    # title
    fig.add_annotation(text=_bold(f'Relative Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Beta'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)
    fig.add_annotation(text=_bold(f'Information Ratio(IR)'), align='left', x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=2)
    fig.add_annotation(text='(Relative return / tracking error)', align='left', x=0, y = 1, yshift=20, xref="x domain", yref="y domain", showarrow=False, font=dict(size=12), row=2, col=2)

    # laoyout
    fig.update_layout(
        legend=dict(x=0.05, y=0.94, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)'),
        width= fig_param['size']['w'],
        height= fig_param['size']['h'],
        margin= fig_param['margin'],
        template= fig_param['template'],
        )
    
    return fig

# multi Report
def plot_efficiency_frontier(returns:pd.DataFrame):
    from scythe.analysis import create_random_portfolios
    portfolios = create_random_portfolios(returns)
    max_sharpe = portfolios.loc[portfolios['sharpe'].argmax()]
    min_vol = portfolios.loc[portfolios['std_dev'].argmin()]
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(
        x=portfolios['std_dev'],
        y=portfolios['return'],
        mode='markers',
        marker=dict(
            size=5,
            color=portfolios['sharpe'],
            colorscale='PuBu',
            opacity=0.3
        ),
        name='Weights',
        text=portfolios['weights'].astype(str),
        showlegend=False,
    ), row=1, col=1)

    # Add maximum Sharpe ratio point
    fig.add_trace(go.Scatter(
        x=[max_sharpe['std_dev']],
        y=[max_sharpe['return']],
        mode='markers',
        marker=dict(size=10, symbol='diamond'),
        name=f"Max Sharpe: {str(max_sharpe['weights'])}",
        text=[str(max_sharpe['weights'])],
        showlegend=True,
    ), row=1, col=1)

    # Add minimum volatility point
    fig.add_trace(go.Scatter(
        x=[min_vol['std_dev']],
        y=[min_vol['return']],
        mode='markers',
        marker=dict(size=10, symbol='diamond'),
        name=f"Min volatility: {str(min_vol['weights'])}",
        text=[str(min_vol['weights'])],
        showlegend=True
    ), row=1, col=1)
    
    corr_plot = returns.corr()
    corr_plot = corr_plot\
        .mask(np.tril(np.ones(corr_plot.shape)).astype(bool))

    fig.add_trace(go.Heatmap(
        z=corr_plot.values,
        x=corr_plot.columns,
        y=corr_plot.index,
        colorscale='RdBu_r',
        text=corr_plot.map("{:.2f}".format).replace('nan', '').values,
        texttemplate="%{text}",
        showscale=False,
    ), row=2, col=1)
    
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(tickformat=".0%", row=1, col=1)
    
    # position
    fig.update_yaxes(domain=[0.0, 1.0], row=1, col=1)
    fig.update_yaxes(domain=[0.15, 0.45], row=2, col=1)
    fig.update_xaxes(domain=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(domain=[0.7, 1.0], row=2, col=1)

    # title
    fig.add_annotation(text=_bold(f'Efficiency frontier'), x=0, y=1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Corr'), x=0.1, y=1, yshift=20, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

    # Update layout
    fig.update_layout(
        legend=dict(x=0.01, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='normal'),
        xaxis_title='Annualised volatility',
        yaxis_title='Annualised returns',
        width=fig_param['size']['w'],
        height=fig_param['size']['h'],
        margin=fig_param['margin'],
        template=fig_param['template'],
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
    )

    return fig

# factor
def plot_quantile_returns(quantiles_returns:pd.DataFrame, benchmark_return:pd.DataFrame)-> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.05,
    )

    if 'LS' not in quantiles_returns.columns:
        quantiles_returns = quantiles_returns.assign(LS = lambda df: df[df.columns.max()] - df[df.columns.min()])

    c_returns = (1+quantiles_returns).cumprod()-1
    CAGRs = (1 + c_returns.iloc[-1]) ** (240 / len(c_returns)) - 1
    alpha_returns = (1+quantiles_returns.sub(benchmark_return, axis=0)).cumprod()-1
    alpha_CAGRs = (1 + alpha_returns.iloc[-1]) ** (240 / len(alpha_returns)) - 1
    
    num_lines = len(quantiles_returns.drop(columns=['LS']).columns)
    color_scale = sample_colorscale('PuBu', [n / (num_lines) for n in range(num_lines)]) + [fig_param['colors']['Bright Magenta']]
    for column, color in zip((list(quantiles_returns.columns)), color_scale):
        # cumulative return
        fig.add_trace(go.Scatter(
            x=c_returns.index,
            y=c_returns[column].values,
            name=int(column) if isinstance(column, float) else column,
            line=dict(color=color if isinstance(column, float) else fig_param['colors']['Bright Magenta']),
            legendgroup=column,
            showlegend=True,
        ),
        row=1, col=1)

        # CAGRs
        fig.add_annotation(
            x=c_returns.index[-1],
            y=c_returns[column].values[-1], 
            xref="x", 
            yref="y", 
            text=f'{int(column) if isinstance(column, float) else column}: {CAGRs[column]: .2%}',  
            showarrow=False, 
            xanchor="left", 
            yanchor="middle", 
            font=dict(color="white", size=10),
        row=1, col=1)

        # alpha cumulative return
        fig.add_trace(go.Scatter(
            x=alpha_returns.index,
            y=alpha_returns[column].values,
            name=int(column) if isinstance(column, float) else column,
            line=dict(color=color if isinstance(column, float) else fig_param['colors']['Bright Magenta']),
            legendgroup=column,
            showlegend=False,
        ),
        row=2, col=1)

        # alpha CAGRs
        fig.add_annotation(
            x=alpha_returns.index[-1],
            y=alpha_returns[column].values[-1], 
            xref="x", 
            yref="y", 
            text=f'{int(column) if isinstance(column, float) else column}: {alpha_CAGRs[column]: .2%}',  
            showarrow=False, 
            xanchor="left", 
            yanchor="middle", 
            font=dict(color="white", size=10),
        row=2, col=1)
    
    # asjust y as % for equity curve & drawdown
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)

    # position
    fig.update_yaxes(domain=[0.55, 1.0], row=1, col=1)
    fig.update_yaxes(domain=[0.0, 0.45], row=2, col=1)

    # titles
    fig.add_annotation(text=_bold(f'Quantile Return'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Quantile Excess Return'), x=0, y = 1, yshift=10, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

    fig.update_layout(
        legend=dict(x=0.04, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='reversed', font=dict(size=10)),    
        height=fig_param['size']['h'], 
        width=fig_param['size']['w'], 
        margin= fig_param['margin'],
        template=fig_param['template'],
    )
    
    return fig

def plot_icir(ic:pd.Series, ir:pd.Series):
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.05,
    )

    fig.add_trace(go.Scatter(
        x=ic.index, 
        y=ic.values, 
        mode='lines', 
        name='IC', 
        line=dict(color=fig_param['colors']['Bright Orange'], width=1),
        showlegend=False,
        ),
        row=1, col=1)
    fig.add_hline(
        y=ic.mean(),
        line=dict(color="white", width=1),
        line_dash="dash",
        annotation_text=f'avg. IC: {ic.mean() :.2}',
        annotation_position="bottom left",
        annotation_textangle = 0,
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=ir.index, 
        y=ir.values, 
        mode='lines', 
        name='IR', 
        line=dict(color=fig_param['colors']['Bright Orange'], width=1),
        showlegend=False,
        ),
        row=2, col=1)
    fig.add_hline(
        y=ir.mean(),
        line=dict(color="white", width=1),
        line_dash="dash",
        annotation_text=f'avg. IR: {ir.mean() :.2}',
        annotation_position="bottom left",
        annotation_textangle = 0,
        row=2, col=1
    )
    
    # position
    fig.update_yaxes(domain=[0.55, 1.0], row=1, col=1)
    fig.update_yaxes(domain=[0.0, 0.45], row=2, col=1)

    # titles
    fig.add_annotation(text=_bold(f'Information Coefficient(IC)'), x=0, y = 1, yshift=30, xref="x domain", yref="y domain", showarrow=False, row=1, col=1)
    fig.add_annotation(text=_bold(f'Information Ratio(IR)'), x=0, y = 1, yshift=40, xref="x domain", yref="y domain", showarrow=False, row=2, col=1)

    fig.update_layout(
        legend=dict(x=0.04, y=1, xanchor='left', yanchor='top', bgcolor='rgba(0,0,0,0)', traceorder='reversed', font=dict(size=10)),    
        height=fig_param['size']['h'], 
        width=fig_param['size']['w'], 
        margin= fig_param['margin'],
        template=fig_param['template'],
    )

    return fig

def plot_double_sorting_surface(data:pd.DataFrame, items:Union[str, list[str]], names:Tuple[str, str]=None):
    x_vals = sorted(list(set([x[0] for x in data.T.index])))
    y_vals = sorted(list(set([x[1] for x in data.T.index])))
    
    if isinstance(items, str):
        items = [items]
    fig = go.Figure(data=[
        go.Surface(
            x=x_vals, y=y_vals, 
            z=[[data.loc[item].loc[(x,y)] for x in x_vals] for y in y_vals],
            colorscale='Inferno',
            showscale=True,
            name = item
        ) for item in items
    ])

    fig.update_layout(
        title=f"{'/'.join(items)} Surface Map",
        scene = dict(
            xaxis_title=names[0] if names is not None else 'Main Group',
            yaxis_title=names[1] if names is not None else 'Ctrl Group',
            zaxis_title=f"{'/'.join(items)}",
            camera_eye={"x": -2, "y": -2, "z": 1},
        ),
        height=fig_param['size']['h'], 
        width=fig_param['size']['w'], 
        margin= fig_param['margin'],
        template=fig_param['template'],
        showlegend=True,
    )

    return fig

def plot_double_sorting_multi_surface(data:pd.DataFrame, items:Union[str, list[str]]=['CAGR(%)', 'MDD(%)', 'Calmar', 't']):
    if len(items) != 4:
        raise ValueError("items must be a list of 4 items")

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]],
        subplot_titles=[_bold(item) for item in items],
    )

    x_vals = sorted(list(set([x[0] for x in data.T.index])))
    y_vals = sorted(list(set([x[1] for x in data.T.index])))
    
    # Add traces
    for i, item in enumerate(items):
        fig.add_trace(go.Surface(
            x=x_vals, y=y_vals, 
            z=[[data.loc[item].loc[(x,y)] for x in x_vals] for y in y_vals],
            colorscale='Inferno',
            showscale=False,
            name=item,
        ),
        row=i//2+1, col=i%2+1)
    
    # Adjust subplot positions
    fig.update_layout(
        scene1=dict(domain=dict(x=[0, 0.5], y=[0.55, 1.0])),
        scene2=dict(domain=dict(x=[0.55, 1.0], y=[0.55, 1.0])),
        scene3=dict(domain=dict(x=[0, 0.5], y=[0, 0.5])),
        scene4=dict(domain=dict(x=[0.55, 1.0], y=[0, 0.5]))
    )
    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=_bold(f"Surface Maps"),
        **{f'scene{i+1}': dict(
            xaxis_title=dict(text=f'Main Group', font=dict(size=10)),
            yaxis_title=dict(text=f'Ctrl Group', font=dict(size=10)), 
            zaxis_title=dict(text=items[i], font=dict(size=10)),
            camera_eye={"x": -2, "y": -2, "z": 1},
        ) for i in range(4)},
        height=fig_param['size']['h'], 
        width=fig_param['size']['w'], 
        margin= fig_param['margin'],
        template=fig_param['template'],
    )

    return fig