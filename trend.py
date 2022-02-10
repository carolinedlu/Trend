import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st
import monthly_returns_heatmap as mrh
import plotly.graph_objects as go

st.set_page_config(
        page_title='Trend Following',layout='wide')

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.sidebar.image("trendlogo.png")

expander=st.expander("User Inputs (Collapse to hide)",expanded=True)
with expander:
        # ### Setting leverage and transaction cost

        port_nav=st.number_input("Enter current portfolio NAV:",min_value=0,value=100000,help='This is the total amount of cash if portfolio is liquidated today, net of all borrowing. Your broker should be able to provide this information. For example, Interactive Brokers calls this Net Liquidation Value.')

        # In[2]:
        param1, param2, param3=st.columns(3)

        with param1:
            st.subheader("Trade Parameters")
            lev=float(st.number_input('Set the leverage:',value=1.5,min_value=0.1,step=0.1,format="%.1f",help='Above 1x means leverage is employed'))
            comm=st.slider('Set the transaction cost in decimals, eg. 0.1% as 0.001:',min_value=0.0000,max_value=0.0100,value=0.0005,step=0.0001,format="%.4f",help='Total cost of executing a single trade expressed as % of trade value')
            spread=st.slider('Set the borrow spread in decimals, eg. 1% as 0.01:',min_value=0.000,max_value=0.100,value=0.015,step=0.001,format="%.3f",help='Margin lending spread charged by broker on top of reference rate')

        with param2:
            st.subheader("Model Parameters")
            stop=st.slider('Set the stop loss in decimals, eg. 10% as 0.1:',min_value=0.00,max_value=1.00,value=0.10,step=0.01,format="%.2f",help='A smaller stop loss gives better downside protection but results in more whipsaws')
            fast=int(st.number_input('Set the fast moving average in number of days:',step=1,min_value=5,value=100,help='A shorter window is better at capturing trend changes but results in more whipsaws'))
            slow=int(st.number_input('Set the slow moving average in number of days:',step=1,min_value=10,value=200,help='A shorter window is better at capturing trend changes but results in more whipsaws'))

        # ### Preparing list of ticker symbols

        # In[3]:

        table=pd.read_html('https://www.investopedia.com/terms/d/djia.asp')
        df=table[0]

        df.columns=['Company','Symbol','Year Added']

        tickers=list(df['Symbol'])

        # In[6]:
        with param3:
            st.subheader("Simulation Parameters")
            start=st.date_input("Enter simulation start date in YYYY/MM/DD format:",value=dt.date(2005,1,1))
            default_tickers=st.radio(label='Keep Dow Jones Industrial Average as the default stock universe?',options=['Y','N'],help='Trend model is applied on Dow Jones Industrial Average component stocks as a default')
            if default_tickers=='N':
                input_string=st.text_input("Enter at least two stock tickers listed in the US, separated by comma or upload a CSV file below:")
                tickers=input_string.split(",")
                uploaded_file = st.file_uploader("Choose CSV file", help='Tickers should be listed in a single column in the CSV file',type='csv')
                if uploaded_file is not None:
                        df = pd.read_csv(uploaded_file,names=['Symbol'])
                        tickers=list(df['Symbol'])

### Creating launch button

if st.button("Run Trend Following",help='Click button to run model once the user inputs above are fixed'):    
    st.write("Model takes about 30s to run")

# ### Preparing daily USD ON Libor

    libor_start = dt.date(2001,1,3)
    libor_end=dt.datetime.now()
    df=yf.download("DIA",libor_start,libor_end)
    libor_close=df['Close']
    libor=np.array(pd.read_csv('USD ON Libor.csv'))
    usd_libor=list(libor[:,-1])

    for i in range(len(usd_libor),len(libor_close)):
        usd_libor.append(usd_libor[i-1])

    libor=pd.DataFrame(usd_libor,columns=['Libor'],index=libor_close.index)

### Getting daily price data from Yahoo! Finance

    end=dt.datetime.now()

    df=yf.download(tickers,start,end)
    DIA_P=yf.download("DIA",start,end)


# In[9]:


    open=df['Open']
    close=df['Close']
    adj_close=df['Adj Close']
    lastpx=close.iloc[-1].map('{0:.2f}'.format)
    DIA=DIA_P['Adj Close']


# ### Calculating daily adjusted open price

# In[10]:


    adj_open=adj_close/close*open


# ### Calculating fast moving average

# In[11]:


    fast_dma=adj_close.rolling(window=fast).mean().values


# ### Calculating slow moving average

# In[12]:


    slow_dma=adj_close.rolling(window=slow).mean().values

# ### Getting USD ON Libor

    libor=pd.concat([close,libor],axis=1).to_numpy()
    usd_libor=list(libor[:,-1])

    for i in range(len(usd_libor),len(close)):
        usd_libor.append(usd_libor[i-1])

# ### Trend following system

# In[13]:


    equity=[]
    port_value=[]

    shares=np.zeros((len(close.index),len(close.columns)))
    buy_price=np.zeros((len(close.index),len(close.columns)))
    sell_price=np.zeros((len(close.index),len(close.columns)))
    pnl=np.zeros((len(close.index),len(close.columns)))
    trade_value=np.zeros((len(close.index),len(close.columns)))
    trade_pnl=np.zeros((len(close.index),len(close.columns)))
    pos_value=np.zeros((len(close.index),len(close.columns)))
    hold=np.zeros((len(close.index),len(close.columns)))
    stop_loss=np.zeros((len(close.index),len(close.columns)))

    for i in range(len(close.index)):
        for j in range(len(close.columns)):
            if i>0 and hold[i-1,j]==0 and fast_dma[i-1,j]>slow_dma[i-1,j] and adj_close.iloc[i-1,j]>slow_dma[i-1,j] and slow_dma[i-1,j]>slow_dma[i-2,j]:
                hold[i,j]=1
                shares[i,j]=np.floor(equity[i-1]*lev/(len(close.columns))/adj_close.iloc[i-1,j])
                buy_price[i,j]=adj_open.iloc[i,j]
                sell_price[i,j]=np.nan
                pnl[i,j]=shares[i,j]*(adj_close.iloc[i,j]-(buy_price[i,j]*(1+comm)))
                trade_value[i,j]=shares[i,j]*buy_price[i,j]*(1+comm)
                pos_value[i,j]=shares[i,j]*adj_close.iloc[i,j]
                stop_loss[i,j]=buy_price[i,j]*(1-stop)
            elif hold[i-1,j]==0:
                buy_price[i,j]=np.nan
                sell_price[i,j]=np.nan
                stop_loss[i,j]=np.nan
            elif hold[i-1,j]==1:
                if fast_dma[i-1,j]<slow_dma[i-1,j] and adj_close.iloc[i-1,j]<stop_loss[i-1,j]:
                    buy_price[i,j]=np.nan
                    sell_price[i,j]=adj_open.iloc[i,j]
                    stop_loss[i,j]=np.nan
                    pnl[i,j]=shares[i-1,j]*(sell_price[i,j]*(1-comm)-adj_close.iloc[i-1,j])
                    trade_value[i,j]=shares[i-1,j]*sell_price[i,j]*(1-comm)
                    trade_pnl[i,j]=trade_value[i,j]/trade_value[i-1,j]-1
                else:
                    hold[i,j]=1
                    shares[i,j]=shares[i-1,j]
                    buy_price[i,j]=np.nan
                    sell_price[i,j]=np.nan
                    pnl[i,j]=shares[i,j]*(adj_close.iloc[i,j]-adj_close.iloc[i-1,j])
                    trade_value[i,j]=trade_value[i-1,j]
                    pos_value[i,j]=shares[i,j]*adj_close.iloc[i,j]
                    if i==len(close.index)-1:
                        trade_pnl[i,j]=pos_value[i,j]/trade_value[i-1,j]-1
                    if adj_close.iloc[i,j]*(1-stop)>stop_loss[i-1,j]:
                        stop_loss[i,j]=adj_close.iloc[i,j]*(1-stop)
                    else:
                        stop_loss[i,j]=stop_loss[i-1,j]
        if i==0:
            eq=3000000.0
            equity.append(eq)
            pval=0.0
            port_value.append(pval)
        else:
            if port_value[i-1]>equity[i-1]:
                borrow=(port_value[i-1]-equity[i-1])*(spread+usd_libor[i-1])/365
            else:
                borrow=0.0
            eq=equity[i-1]+np.sum(pnl[i,:])-borrow
            equity.append(eq)
            pval=np.sum(pos_value[i,:])
            port_value.append(pval)

# In[14]:

# ### Plotting leverage chart

    p_lev=[]
    for i in range(len(equity)):
        leverage=port_value[i]/equity[i]
        p_lev.append(leverage)
    current_lev=p_lev[len(p_lev)-1]
    st.write('### The current portfolio leverage is', "{:.2f}".format(current_lev))
    data={'Leverage':p_lev}
    p_lev=pd.DataFrame(data,index=close.index)

    col1,col2=st.columns(2)
    with col2:
            fig=plt.figure(figsize=(15,7))
            plt.title('Portfolio Leverage',fontsize=28)
            plt.plot(p_lev)
            x=[close.index[0],close.index[-1]]
            y=[1,1]
            plt.plot(x,y)
            st.pyplot(fig)

# ### Plotting NAV curve

    tf_nav=[]
    tf_dd=[]

    for i in range(len(equity)):
        if i==0:
            nav=1.0
            tf_nav.append(nav)
            dd=0.0
            tf_dd.append(dd)
        else:
            nav=equity[i]/equity[i-1]*tf_nav[i-1]
            tf_nav.append(nav)
            dd=equity[i]/max(equity[0:i])-1
            tf_dd.append(dd)
    data={'Trend Following':tf_nav}
    tf_curve=pd.DataFrame(data,index=close.index)

    with col1:
            if default_tickers=='Y':
                bh_nav=[]
                bh_dd=[]

                for i in range(len(equity)):
                    if i==0:
                        nav=1.0
                        bh_nav.append(nav)
                        dd=0.0
                        bh_dd.append(dd)
                    else:
                        nav=DIA[i]/DIA[i-1]*bh_nav[i-1]
                        bh_nav.append(nav)
                        dd=DIA[i]/max(DIA[0:i])-1
                        bh_dd.append(dd)
                data2={'DIA':bh_nav}
                bh_curve=pd.DataFrame(data2,index=close.index)

                fig=plt.figure(figsize=(15,7))
                plt.title('Net Asset Value',fontsize=28)
                plt.plot(tf_curve,label='Trend Following')
                plt.plot(bh_curve,label='DIA')
                plt.legend(fontsize=18)
                st.pyplot(fig)
            else:
                fig=plt.figure(figsize=(15,7))
                plt.title('Net Asset Value',fontsize=28)
                plt.plot(tf_curve,label='Trend Following')
                plt.legend(fontsize=18)
                st.pyplot(fig)

# ### Trend Following monthly performance table

# In[16]:

    returns=tf_curve.pct_change()
# pip install monthly-returns-heatmap
    fig=mrh.plot(returns,figsize=(15,7),eoy=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

# ### Calculating daily log return

# In[17]:

    if default_tickers=='Y':
        tf_logret=np.log(tf_curve/tf_curve.shift(1))
        bh_logret=np.log(bh_curve/bh_curve.shift(1))
        logret=pd.concat([tf_logret,bh_logret],axis=1)
    else:
        logret=np.log(tf_curve/tf_curve.shift(1))

# ### Calculating performance metrics

# In[18]:

    totalret=np.exp(logret.describe().loc['mean']*logret.describe().loc['count'])-1
    cagr=np.exp(logret.describe().loc['mean']*252)-1
    vol=logret.describe().loc['std']*np.sqrt(252)
    sharpe=cagr/vol
    max_tf=np.max(np.max(mrh.get(returns)))
    min_tf=np.min(np.min(mrh.get(returns)))
    perc_pos_tf = np.count_nonzero(mrh.get(returns).fillna(0) > 0)/np.count_nonzero(mrh.get(returns).fillna(0))
    if default_tickers=='Y':
        max_bh=np.max(np.max(mrh.get(bh_curve.pct_change())))
        min_bh=np.min(np.min(mrh.get(bh_curve.pct_change())))
        perc_pos_bh = np.count_nonzero(mrh.get(bh_curve.pct_change()).fillna(0) > 0)/np.count_nonzero(mrh.get(bh_curve.pct_change()).fillna(0))
        data={'Total Return':totalret,'CAGR':cagr,'Annualized Volatilty':vol,'Sharpe Ratio':sharpe,'Max Drawdown':[min(tf_dd),min(bh_dd)],'Max Monthly Return':[max_tf,max_bh],'Min Monthly Return':[min_tf,min_bh],'% Positive Months':[perc_pos_tf,perc_pos_bh]}
    else:
        data={'Total Return':totalret,'CAGR':cagr,'Annualized Volatilty':vol,'Sharpe Ratio':sharpe,'Max Drawdown':min(tf_dd),'Max Monthly Return':max_tf,'Min Monthly Return':min_tf,'% Positive Months':perc_pos_tf}
    metric=pd.DataFrame(data).T
    metric.iloc[:3]=metric.iloc[:3].applymap('{0:.2%}'.format)
    metric.iloc[3:4]=metric.iloc[3:4].applymap('{0:.2f}'.format)
    metric.iloc[4:8]=metric.iloc[4:8].applymap('{0:.2%}'.format)
    expander=st.expander("Performance Statistics (Click to expand)")
    with expander:
        st.write(metric)

# ### Portfolio status and exposure

# In[20]:

    st.sidebar.header("Trade Alert")
    p_status=[]
    p_exp=[]
    l_price=[]
    shares=[]

    for j in range(len(close.columns)):
        if hold[-2,j]==0 and fast_dma[-1,j]>slow_dma[-1,j] and adj_close.iloc[-1,j]>slow_dma[-1,j] and slow_dma[-1,j]>slow_dma[-2,j]:
            status='BUY'
            p_status.append(status)
            st.sidebar.write("Buy", close.columns[j])
            exp='{:.2%}'.format(lev/(len(close.columns)))
            p_exp.append(exp)
            price=lastpx[j]
            l_price.append(price)
            unit='{:.2f}'.format(port_nav*(pos_value[-1,j]/equity[-1])/float(price))
            shares.append(unit)
        elif hold[-2,j]==1:
            if hold[-1,j]==0:
                status=''
                p_status.append(status)
                exp=''
                p_exp.append(exp)
                price=''
                l_price.append(price)
                unit=''
                shares.append(unit)
            elif fast_dma[-1,j]<slow_dma[-1,j] and adj_close.iloc[-1,j]<stop_loss[-1,j]:
                status='SELL'
                p_status.append(status)
                st.sidebar.write("Sell", close.columns[j])
                exp=''
                p_exp.append(exp)
                price=lastpx[j]
                l_price.append(price)
                unit=''
                shares.append(unit)
            else:
                status='HOLD'
                p_status.append(status)
                exp='{:.2%}'.format(pos_value[-1,j]/equity[-1])
                p_exp.append(exp)
                price=lastpx[j]
                l_price.append(price)
                unit='{:.2f}'.format(port_nav*(pos_value[-1,j]/equity[-1])/float(price))
                shares.append(unit)
        else:
            status=''
            p_status.append(status)
            exp=''
            p_exp.append(exp)
            price=''
            l_price.append(price)
            unit=''
            shares.append(unit)

    p_status=pd.DataFrame(p_status,index=close.columns)
    p_exp=pd.DataFrame(p_exp,index=close.columns)
    l_price=pd.DataFrame(l_price,index=close.columns)
    shares=pd.DataFrame(shares,index=close.columns)
    df=pd.concat([p_status,p_exp,l_price,shares],axis=1)
    df.columns=['Status','Exposure','Last Price','Shares']
    st.sidebar.header("Portfolio Status")
    st.sidebar.markdown('*HOLD - Stock is currently held in portfolio')
    st.sidebar.markdown('*BUY - Add stock to portfolio')
    st.sidebar.markdown('*SELL - Close out stock')
    st.sidebar.table(df)

# ### Calculating profit loss ratio

# In[19]:

    win=np.sum(trade_pnl>0,axis=0)
    total_win=np.sum(win)
    win=np.append(win,total_win)
    lose=np.sum(trade_pnl<0,axis=0)
    total_lose=np.sum(lose)
    lose=np.append(lose,total_lose)
    win_rate=win/(win+lose)
    total_p=[]
    total_l=[]

    for j in range(len(tickers)):
        p=0
        for i in range(len(trade_pnl)):
            if trade_pnl[i,j]>0:
                p=trade_pnl[i,j]+p
        total_p.append(p)

    for j in range(len(tickers)):
        l=0
        for i in range(len(trade_pnl)):
            if trade_pnl[i,j]<0:
                l=trade_pnl[i,j]+l
        total_l.append(l)

    overall_p=np.sum(total_p)
    total_p=np.append(total_p,overall_p)
    overall_l=np.sum(total_l)
    total_l=np.append(total_l,overall_l)
    average_p=total_p/win
    average_l=total_l/lose
    pl_ratio=-average_p/average_l
    ticker=pd.DataFrame(close.columns)
    ticker.loc[len(tickers)]='Total'
    data={'Winning Trades':win,'Losing Trades':lose,'Win %':win_rate,'Average % Gain Per Winning Trade':average_p,'Average % Loss Per Losing Trade':average_l,'Profit/Loss Ratio':pl_ratio}
    df=pd.DataFrame(data)
    pl=pd.concat([ticker,df],axis=1)

    expander=st.expander("Trade Statistics (Click to expand)")
    with expander:
        fig = go.Figure(data=[go.Table(
            columnwidth=[1,1.5,1.5,1,2.5,2.5,2],
            header=dict(values=['','<b>Winning Trades</b>','<b>Losing Trades','<b>Win %','<b>Average % Gain Per Winning Trade','<b>Average % Loss Per Losing Trade','<b>Profit/Loss Ratio'],
                        fill_color='royalblue',
                        font=dict(family="Arial",size=14,color='white'),
                        align='center'),
            cells=dict(values=pl.transpose().values.tolist(),
                       fill=dict(color=['royalblue','white','white','white','white','white','paleturquoise']),
                       font=dict(family="Arial",size=12,color=['white','black','black','black','black','black','black']),
                       align='center',format=["","","",".2%",".2%",".2%",".2f"]))])        
        fig.update_layout(margin=dict(l=5,r=5,b=5,t=5),paper_bgcolor='white')
        st.write(fig)

st.sidebar.subheader('Disclaimer:')
st.sidebar.write('We are not financial advisors or fund managers. This tool is provided purely for informational purposes only. It is not intended to be, nor shall it be construed as, financial advice, an offer, or a solicitation of an offer, to buy or sell an interest in any investment product.Nothing in this tool constitutes investment, accounting, regulatory, tax or other advice. While the information, text, graphics, links and other items provided in this tool is believed to be reliable, AllQuant make no representation or warranty, whether expressed or implied, and accept no responsibility for its completeness, accuracy or reliability. AllQuant also accepts no liability whatsoever with respect to the use of the content in this tool, whether directly or indirectly. All investments are subject to investment risks including the possible loss of the principal amount invested. Any performance shown, whether actual historical, hypothetical or modelled, is not necessarily indicative nor a guarantee on future performance and should not be the sole factor of consideration when investing. You should make your own assessment of the relevance,accuracy and adequacy of the information contained in this tool and consult your independent advisors where necessary.')
