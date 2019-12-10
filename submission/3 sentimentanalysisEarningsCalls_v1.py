#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:56:44 2019

@author: lindaxju
"""

#%%
import sys
sys.setrecursionlimit(100000)

import pickle

import pandas as pd
import numpy as np

from datetime import datetime

from nltk.stem import WordNetLemmatizer
#%%
###############################################################################
##############################Helper Functions#################################
###############################################################################
#%%
###############################Get Sentiment###################################
#%%
def get_sentiment_df(company_transcripts_df):
            
    """
    Input is dataframe of company transcripts
    Returns dataframe of counts and percentage of sentiment words
    """
    
    word_count_all = []
    negative_words_all = []
    positive_words_all = []
    uncertainty_words_all = []
    litigious_words_all = []
    constraining_words_all = []
    interesting_words_all = []
    modal1_words_all = []
    modal2_words_all = []
    modal3_words_all = []
    
    negative_count_all = []
    positive_count_all = []
    uncertainty_count_all = []
    litigious_count_all = []
    constraining_count_all = []
    interesting_count_all = []
    modal1_count_all = []
    modal2_count_all = []
    modal3_count_all = []
    
    for transcript in company_transcripts_df.content_cleaned:
        
        negative_words = []
        positive_words = []
        uncertainty_words = []
        litigious_words = []
        constraining_words = []
        interesting_words = []
        modal1_words = []
        modal2_words = []
        modal3_words = []
    
        prev_word = np.nan
    
        for word in transcript.split():
            
            if word in negative_list:
                negative_words.append(word)
            elif prev_word in negation_list and word in positive_list:
                negative_words.append(prev_word+' '+word)
            elif word in positive_list:
                positive_words.append(word)
            
            if word in uncertainty_list:
                uncertainty_words.append(word)
            
            if word in litigious_list:
                litigious_words.append(word)
            
            if word in constraining_list:
                constraining_words.append(word)
                
            if word in interesting_list:
                interesting_words.append(word)
            
            if word in modal1_list:
                modal1_words.append(word)
            
            if word in modal2_list:
                modal2_words.append(word)
            
            if word in modal3_list:
                modal3_words.append(word)
            
            prev_word = word
                
        negative_words_all.append(negative_words)
        positive_words_all.append(positive_words)
        uncertainty_words_all.append(uncertainty_words)
        litigious_words_all.append(litigious_words)
        constraining_words_all.append(constraining_words)
        interesting_words_all.append(interesting_words)
        modal1_words_all.append(modal1_words)
        modal2_words_all.append(modal2_words)
        modal3_words_all.append(modal3_words)
        
        word_count_all.append(len(transcript))
        negative_count_all.append(len(negative_words))
        positive_count_all.append(len(positive_words))
        uncertainty_count_all.append(len(uncertainty_words))
        litigious_count_all.append(len(litigious_words))
        constraining_count_all.append(len(constraining_words))
        interesting_count_all.append(len(interesting_words))
        modal1_count_all.append(len(modal1_words))
        modal2_count_all.append(len(modal2_words))
        modal3_count_all.append(len(modal3_words))
    
    company_transcripts_sentiment_df = company_transcripts_df.copy()
    company_transcripts_sentiment_df['words_negative'] = negative_words_all
    company_transcripts_sentiment_df['words_positive'] = positive_words_all
    company_transcripts_sentiment_df['words_uncertainty'] = uncertainty_words_all
    company_transcripts_sentiment_df['words_litigious'] = litigious_words_all
    company_transcripts_sentiment_df['words_constraining'] = constraining_words_all
    company_transcripts_sentiment_df['words_interesting'] = interesting_words_all
    company_transcripts_sentiment_df['words_modal1'] = modal1_words_all
    company_transcripts_sentiment_df['words_modal2'] = modal2_words_all
    company_transcripts_sentiment_df['words_modal3'] = modal3_words_all
    
    company_transcripts_sentiment_df['count_allwords'] = word_count_all
    company_transcripts_sentiment_df['count_negative'] = negative_count_all
    company_transcripts_sentiment_df['per_negative'] = [i/j*100 for i, j in zip(negative_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_positive'] = positive_count_all
    company_transcripts_sentiment_df['per_positive'] = [i/j*100 for i, j in zip(positive_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_uncertainty'] = uncertainty_count_all
    company_transcripts_sentiment_df['per_uncertainty'] = [i/j*100 for i, j in zip(uncertainty_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_litigious'] = litigious_count_all
    company_transcripts_sentiment_df['per_litigious'] = [i/j*100 for i, j in zip(litigious_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_constraining'] = constraining_count_all
    company_transcripts_sentiment_df['per_constraining'] = [i/j*100 for i, j in zip(constraining_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_interesting'] = interesting_count_all
    company_transcripts_sentiment_df['per_interesting'] = [i/j*100 for i, j in zip(interesting_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_modal1'] = modal1_count_all
    company_transcripts_sentiment_df['per_modal1'] = [i/j*100 for i, j in zip(modal1_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_modal2'] = modal2_count_all
    company_transcripts_sentiment_df['per_modal2'] = [i/j*100 for i, j in zip(modal2_count_all, word_count_all)]
    company_transcripts_sentiment_df['count_modal3'] = modal3_count_all
    company_transcripts_sentiment_df['per_modal3'] = [i/j*100 for i, j in zip(modal3_count_all, word_count_all)]
    
    return company_transcripts_sentiment_df
#%%
################################Sort in Order##################################
#%%
def sort_company_df(df_to_be_sorted):
                
    """
    Input is dataframe to be sorted with transcript names including dates
    Returns dataframe sorted by dates
    """
    
    transcript_dates = []
    for transcript_name in df_to_be_sorted.transcript_name:
        transcript_dates.append(datetime.strptime(transcript_name[1], '%m/%d/%y'))
    
    df_sorted = df_to_be_sorted.copy()
    df_sorted.insert(loc=0,column='date',value=transcript_dates)
    
    df_sorted.sort_values(by=['date'],inplace=True)
    df_sorted.reset_index(drop=True,inplace=True)
    
    return df_sorted
#%%
###############################################################################
##########################Get Sentiment Over Time##############################
###############################################################################
#%%
##########################Import Cleaned Transcripts###########################
#%%
filename = 'data/' + 'company_df_2019-12-07-16-01-29.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
#########################Import Financial Dictionary###########################
#%%
lm_df = pd.read_csv('data/LoughranMcDonald_MasterDictionary_2018.csv')
lm_df.head()
#%%
negation_list = ['no', 'not', 'none', 'neither', 'never', 'nobody']
negative_list = []
positive_list = []
uncertainty_list = []
litigious_list = []
constraining_list = []
interesting_list = []
modal1_list = []
modal2_list = []
modal3_list = []

lemmatizer = WordNetLemmatizer()

for i in lm_df.index:
    
    word = lemmatizer.lemmatize(str(lm_df.Word[i]).lower())
    
    if lm_df.Negative[i] != 0:
        negative_list.append(word)
    elif lm_df.Positive[i] != 0:
        positive_list.append(word)
    
    if lm_df.Uncertainty[i] != 0:
        uncertainty_list.append(word)
        
    if lm_df.Litigious[i] != 0:
        litigious_list.append(word)
        
    if lm_df.Constraining[i] != 0:
        constraining_list.append(word)
        
    if lm_df.Interesting[i] != 0:
        interesting_list.append(word)
        
    if lm_df.Modal[i] != 0:
        if lm_df.Modal[i] == 1:
            modal1_list.append(word)
        elif lm_df.Modal[i] == 2:
            modal2_list.append(word)
        else:
            modal3_list.append(word)
#%%
#########################Perform Sentiment Analysis############################
#%%
# combine all paragraphs into one per transcript
company_df_temp1 = company_df.groupby(['transcript_name'])['content_orig'].apply(' '.join).reset_index()
company_df_temp2 = company_df.groupby(['transcript_name'])['content_cleaned'].apply(' '.join).reset_index()
company_transcripts_df = pd.merge(company_df_temp1,company_df_temp2,on='transcript_name',how='left')
company_transcripts_df
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_transcripts_df'+'_'+timestamp
#company_transcripts_df.to_csv(r'data/'+filename+'.csv')
#%%
company_sentiment_df = get_sentiment_df(company_transcripts_df)
company_sentiment_df
#%%
company_sentiment_ordered_df = sort_company_df(company_sentiment_df)
company_sentiment_ordered_df
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_sentiment_ordered_df'+'_'+timestamp
#company_sentiment_ordered_df.to_csv(r'data/'+filename+'.csv')
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_sentiment_ordered_df'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(company_sentiment_ordered_df, to_write)
#%%
###############################################################################
####################################End########################################
###############################################################################
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
test_phrase = "operator everyone thanks joining conference call discus second result full year outlook call today gennette chairman ceo price cfo several prepared remark share open session operator instruction addition call press release posted slide investor section website macysinc com summarizes information prepared remark well additional fact figure regarding operating performance guidance additionally form filed week available website time one administrative note share adjusting timing typical quarterly earnings release beginning next quarterly earnings call thursday november rough rule thumb barring minor conflict going forward plan release result first third third thursday end however timing fourth call not planned change addition publishing earnings release hour earlier moving time call eastern time quarterly result please mark calendar accordingly keep mind forward looking statement subject safe harbor provision private security litigation reform act forward looking statement subject risk uncertainty could cause actual result differ materially expectation assumption mentioned today detailed factor uncertainty contained company filing security exchange commission discussing result operation providing adjusted net income diluted earnings per share amount exclude impact impairment cost find additional information regarding non gaap financial measure well others used earnings release call investor section website today call webcast website replay available hour conclusion call archived following call year take second result open line saw press release macy inc delivered another comparable sale growth achieved increase comparable sale owned plus licensed basis earnings per share wa well expectation communicated first earnings call walked second elevated spring inventory second got slow start moved deeper became clear inventory wa mounting problem due combination factor miss fashion key woman sportswear private brand slow sell warm weather apparel accelerated decline international tourism business took necessary markdowns clear inventory unanticipated markdowns impacted gross margin second almost full point enter fall season right inventory level mix meet anticipated customer demand improved freshness liquidity respond season fall trend remain confident annual comparable sale guidance flat said light second result lowering adjusted earnings guidance full year expect adjusted earnings per share range including asset sale gain minus asset sale gain step back look business broadly seasonal inventory challenge second many area business performing well brick mortar business healthier led growth store backstage expansion digital business delivered th consecutive double digit growth make good progress strategic initiative track expected contribute comparable sale growth back half year quickly review growth additional macy store receiving growth treatment work store complete ahead last year growth rollout well advance holiday season continue see strong performance original growth store expect similar sale lift one work completed store make macy brick mortar sale backstage expanding backstage another store already complete backstage within macy across country encouraging see backstage location within macy store open month continue comp mid single digit getting better price every day backstage distribution center columbus ohio running dc operates google cloud platform giving improved efficiency speed scale support continued growth price business part supply chain strategy roll google cloud platform distribution center providing additional network flexibility speed macy bloomingdale third vendor direct continue aggressively grow skus brand able offer customer macys com vendor direct halfway goal adding skus year customer love expanded selection vendor direct ha upside add sale gross margin increase customer satisfaction traffic site capital inventory investment make high roic rate mobile mobile help create better omnichannel experience customer enabling shop easily frequently store online app customer valued assistant interacting macy bloomingdale brand continue see significant increase usage conversion mobile remains fastest growing channel regular cadence new feature wallet store stylist many improve store customer experience like quick barcode pickup online order localized product recommendation store reward exclusive app lastly destination business destination business group second strongest performance fine jewelry men tailored woman shoe destination business account nearly total sale macy higher aur business continue outperform balance business market share return investment profitability putting additional resource behind category drive growth great product top performing colleague improved environment enhanced marketing destination point entry macy brand see significant opportunity drive cross shopping one competitive advantage department store pleased market share growth business seen date wo not satisfied macy taking market share overall get business business starting initiative team hard work implementing confident important contributor back half performance bloomingdale challenging driven largely decline international tourism woman men shoe bloomingdale outlet standouts third going opening additional outlet store bluemercury another strong opened new store total store across state bluemercury freestanding store shop within macy store grew sale shop shop showing considerable growth bluemercury com continues significant runway growing double digit continue innovate across channel strengthen relationship existing customer bring new customer brand july launched second iteration story macy outdoor story pleased customer response far story brings new customer give current customer another reason visit often also open door new partnership local niche brand well major player outdoor story partnered dick sporting goods scott miracle gro collaboration allowed offer customer unique product experience also testing opportunity recommerce rental tap changing customer behavior especially among millennial gen consumer month began pilot thredup world largest fashion resale marketplace macy door across country know many consumer passionate sustainable fashion shopping resale partnership give opportunity reach new customer keep coming back shop ever changing selection style brand not typically carry relates fashion rental know customer want variety discovery tapping bloomingdale launching list subscription rental service partner caastle learning bloomingdale inform development similar rental service macy near future look ahead second half confident right plan place continue grow business second gross margin wa tough inventory right level laid holiday must win entire organization leverage need competitive today environment consumer spending remains healthy significant noise macroeconomy tariff currency fluctuation declining international tourism name dynamic situation team prepared respond change consumer environment relates specifically tariff currently assessing detail trade representative office released yesterday related fourth tranche tariff goods imported china confidence scale give leverage find mitigation strategy work vendor partner supplier china know earlier tranche tariff though today customer doe not much appetite price increase closing first half ha challenging proud work team ha done drive business dynamic external environment course correct needed committed delivering back half year remain focused continued top line growth market share growth growth customer base importantly also clear line sight profitability growth hard work productivity initiative touch shortly share detail together early september know negative sentiment sector confident plan strong consumer demand high quality affordable fashion strengthening relationship current customer bringing new customer brand today consumer demand ability shop anytime anywhere know deliver convenience omnichannel retailer stabilized business strong aligned team balance sheet healthy resource invest create long term shareholder value going hand review second provide detail outlook year achieved another consecutive positive comparable sale not pleased overall performance said fully focused successful fall season confident benefit expect deliver second half year strategic initiative funding future productivity program second delivered sale increase owned plus licensed comparable basis mentioned saw strength within destination business experienced softness home category ready wear continued challenge saw strongest performance north central region digital continued deliver solid growth international tourism sale accelerating headwind sequentially encouraging see growth total transaction continues key driver positive sale comp average unit per transaction platinum customer purchased fewer unit average transact importantly continue spend aggregate average unit retail wa driven growth backstage heightened markdowns well challenging comparison strong aur performance second generated credit revenue last year line expectation star reward loyalty program continues drive good momentum credit revenue credit card penetration basis point however slight uptick fraud bad debt monitor metric closely strategy place mitigate exposure balance year remain confident annual credit revenue guidance range remains unchanged gross margin second wa basis point last year combined additional markdowns mentioned growth delivery drove vast majority decline increased delivery cost supporting online growth star reward program anticipated impact margin wa roughly equal saw first however wa additional markdowns largely responsible much steeper decline gross margin expected start inventory overhang start tougher sale environment anticipated team took necessary markdowns clear excess spring inventory markdowns resulted significantly lower gross margin ended comp inventory nearly flat versus first taking markdowns wa certainly tough medicine wa important enter fall season aligned inventory sale plan achieved confident plan deliver normal level markdowns back half year confidence protecting margin supported several factor first inventory well positioned fall receipt planning much leaner allowing maintain liquidity use opportunistically customer demand dictate second majority unanticipated second markdowns related woman sportswear private brand much improved inventory position managed much greater discipline problem control put new leadership place deep examination aspect business take time get sportswear business growing healthily area ready wear third strategic inventory allocation marketing example continue challenge status quo drive effectiveness efficiency medium le includes simplifying offer leveraging strategic partnership improving segmentation targeting customer already taken several action simplify eliminate le effective marketing promotion saving incremental ebit annualized basis fourth making good progress productivity initiative one initiative discussed many time hold flow enables dynamically reallocate inventory season based need resulting fewer markdowns stock initial result test encouraging found hold flow product average generated incremental margin cost net benefit could translate ten incremental ebit annualized basis give great confidence operating program scale fall expect see significant ebit benefit result additionally enhancing data analytics capability getting granular understanding pricing markdown decision also enhance margin rolling new pricing capability scale following successful spring test also sizable ebit benefit expected moving sg recorded expense increase basis point rate basis last year increase sg wa primarily driven investment strategic initiative given front loaded nature investment continue expect growth sg year year weighted towards spring season benefit restructuring normal ongoing cost reduction previously discussed weighted towards fall season interest expense continued benefit lower debt level balance sheet remains healthy effective tax rate wa second expect year variance caused certain normal discrete tax benefit occur unevenly year summing delivered adjusted net income versus last year included net income figure asset sale gain respectively adjusted eps wa compared last year asset sale gain represented respectively year date cash flow operating activity wa compared last year variance due lower merchandise payable adjusted ebitda offset lower tax payment capital expenditure compared last year remain track achieve existing guidance capital expenditure cash used financing activity wa le year ago paid le debt year last year guidance full year continue expect flat total sale growth comp flat owned plus licensed basis confident deliver upper end sale range full strength strategic initiative learned challenge holiday season consumer spending continues healthy although noted mindful macro uncertainty full guidance range contemplates looking fall season sale specifically expect fourth comp meaningfully greater third comp third cycling toughest comp sale comparison year large part due benefit saw cooler temperature last october flip side fourth cycling disruption caused fire west virginia mega center underperformance pre christmas promotional event gross margin fall season expected slightly year ago confidence trend improvement based factor mentioned earlier inventory parity merchant team liquidity enhanced precision fall promotion productivity program regarding inventory expecting rise end third intentionally bringing receipt support early november sale truncated holiday season still expect last year end fall season look asset sale gain planned back half year anticipate balance guidance occur fourth expect adjusted eps range excluding asset sale gain potential impact fourth tranche tariff not contemplated guidance still processing detail released yesterday noted active vendor supplier mitigate tariff minimize customer impact much possible know coming week respect overall guidance confident outcome control also mindful cannot control providing guidance prudent range cognizant current macro uncertainty find complete guidance slide posted website earlier commit sharing detail funding future productivity program early september along lawton president lay productivity strategy fully including detail work stream anticipated total saving next year wrap second wa indeed challenge confident right track accomplish short term long term objective strengthened balance sheet steady debt retirement healthy position allows additional flexibility making right investment business hyperfocused profitable growth making good progress initiative enhance customer shareholder value strategic initiative growing sale productivity program transform way work diligent inventory management capital allocation top mind day day improving management discipline every day data analytics better tool better process strong team working together win every day"
#%%
print(len(negative_list))
print(len(positive_list))
print(len(uncertainty_list))
print(len(litigious_list))
print(len(constraining_list))
print(len(interesting_list))
print(len(modal1_list))
print(len(modal2_list))
print(len(modal3_list))
#%%
test_phrase
#%%
test_phrase_list = test_phrase.split()
#%%
len(test_phrase_list)
#%%
negative_count = []
positive_count = []
uncertainty_count = []
litigious_count = []
constraining_count = []
interesting_count = []
modal1_count = []
modal2_count = []
modal3_count = []

last_word = np.nan

for word in test_phrase_list:
    if word in negative_list:
        negative_count.append(word)
#    elif word in positive_list and last_word not in negation_list:
    elif last_word in negation_list and word in positive_list:
        negative_count.append(last_word+' '+word)
    elif word in positive_list:
        positive_count.append(word)
    
    if word in uncertainty_list:
        uncertainty_count.append(word)
    
    if word in litigious_list:
        litigious_count.append(word)
    
    if word in constraining_list:
        constraining_count.append(word)
        
    if word in interesting_list:
        interesting_count.append(word)
    
    if word in modal1_list:
        modal1_count.append(word)
    
    if word in modal2_list:
        modal2_count.append(word)
    
    if word in modal3_list:
        modal3_count.append(word)
    
    last_word = word
#%%
print(negative_count)
print(positive_count)
print(uncertainty_count)
print(litigious_count)
print(constraining_count)
print(interesting_count)
print(modal1_count)
print(modal2_count)
print(modal3_count)
#%%
#%%
word_count_all = []
negative_words_all = []
positive_words_all = []
uncertainty_words_all = []
litigious_words_all = []
constraining_words_all = []
interesting_words_all = []
modal1_words_all = []
modal2_words_all = []
modal3_words_all = []

negative_count_all = []
positive_count_all = []
uncertainty_count_all = []
litigious_count_all = []
constraining_count_all = []
interesting_count_all = []
modal1_count_all = []
modal2_count_all = []
modal3_count_all = []

for transcript in company_transcripts_df.content_cleaned:
    
    negative_words = []
    positive_words = []
    uncertainty_words = []
    litigious_words = []
    constraining_words = []
    interesting_words = []
    modal1_words = []
    modal2_words = []
    modal3_words = []

    prev_word = np.nan

    for word in transcript.split():
        
        if word in negative_list:
            negative_words.append(word)
        elif prev_word in negation_list and word in positive_list:
            negative_words.append(prev_word+' '+word)
        elif word in positive_list:
            positive_words.append(word)
        
        if word in uncertainty_list:
            uncertainty_words.append(word)
        
        if word in litigious_list:
            litigious_words.append(word)
        
        if word in constraining_list:
            constraining_words.append(word)
            
        if word in interesting_list:
            interesting_words.append(word)
        
        if word in modal1_list:
            modal1_words.append(word)
        
        if word in modal2_list:
            modal2_words.append(word)
        
        if word in modal3_list:
            modal3_words.append(word)
        
        prev_word = word
            
    negative_words_all.append(negative_words)
    positive_words_all.append(positive_words)
    uncertainty_words_all.append(uncertainty_words)
    litigious_words_all.append(litigious_words)
    constraining_words_all.append(constraining_words)
    interesting_words_all.append(interesting_words)
    modal1_words_all.append(modal1_words)
    modal2_words_all.append(modal2_words)
    modal3_words_all.append(modal3_words)
    
    word_count_all.append(len(transcript))
    negative_count_all.append(len(negative_words))
    positive_count_all.append(len(positive_words))
    uncertainty_count_all.append(len(uncertainty_words))
    litigious_count_all.append(len(litigious_words))
    constraining_count_all.append(len(constraining_words))
    interesting_count_all.append(len(interesting_words))
    modal1_count_all.append(len(modal1_words))
    modal2_count_all.append(len(modal2_words))
    modal3_count_all.append(len(modal3_words))
#%%
company_transcripts_sentiment_df = company_transcripts_df.copy()
company_transcripts_sentiment_df['words_negative'] = negative_words_all
company_transcripts_sentiment_df['words_positive'] = positive_words_all
company_transcripts_sentiment_df['words_uncertainty'] = uncertainty_words_all
company_transcripts_sentiment_df['words_litigious'] = litigious_words_all
company_transcripts_sentiment_df['words_constraining'] = constraining_words_all
company_transcripts_sentiment_df['words_interesting'] = interesting_words_all
company_transcripts_sentiment_df['words_modal1'] = modal1_words_all
company_transcripts_sentiment_df['words_modal2'] = modal2_words_all
company_transcripts_sentiment_df['words_modal3'] = modal3_words_all

company_transcripts_sentiment_df['count_allwords'] = word_count_all
company_transcripts_sentiment_df['count_negative'] = negative_count_all
company_transcripts_sentiment_df['per_negative'] = [i/j*100 for i, j in zip(negative_count_all, word_count_all)]
company_transcripts_sentiment_df['count_positive'] = positive_count_all
company_transcripts_sentiment_df['per_positive'] = [i/j*100 for i, j in zip(positive_count_all, word_count_all)]
company_transcripts_sentiment_df['count_uncertainty'] = uncertainty_count_all
company_transcripts_sentiment_df['per_uncertainty'] = [i/j*100 for i, j in zip(uncertainty_count_all, word_count_all)]
company_transcripts_sentiment_df['count_litigious'] = litigious_count_all
company_transcripts_sentiment_df['per_litigious'] = [i/j*100 for i, j in zip(litigious_count_all, word_count_all)]
company_transcripts_sentiment_df['count_constraining'] = constraining_count_all
company_transcripts_sentiment_df['per_constraining'] = [i/j*100 for i, j in zip(constraining_count_all, word_count_all)]
company_transcripts_sentiment_df['count_interesting'] = interesting_count_all
company_transcripts_sentiment_df['per_interesting'] = [i/j*100 for i, j in zip(interesting_count_all, word_count_all)]
company_transcripts_sentiment_df['count_modal1'] = modal1_count_all
company_transcripts_sentiment_df['per_modal1'] = [i/j*100 for i, j in zip(modal1_count_all, word_count_all)]
company_transcripts_sentiment_df['count_modal2'] = modal2_count_all
company_transcripts_sentiment_df['per_modal2'] = [i/j*100 for i, j in zip(modal2_count_all, word_count_all)]
company_transcripts_sentiment_df['count_modal3'] = modal3_count_all
company_transcripts_sentiment_df['per_modal3'] = [i/j*100 for i, j in zip(modal3_count_all, word_count_all)]
#%%
company_transcripts_ordered = list(reverse(company_df.transcript_name.unique()))
company_transcripts_ordered
#%%
transcript_name_index = []
for transcript_name in company_topics_sum_df.transcript_name:
    transcript_name_index.append(company_transcripts_ordered.index(transcript_name))
#%%
company_topics_ordered_df = company_topics_sum_df.copy()
company_topics_ordered_df['transcript_name_index'] = transcript_name_index
company_topics_ordered_df.sort_values(by=['transcript_name_index'],inplace=True)
company_topics_ordered_df.reset_index(drop=True,inplace=True)
company_topics_ordered_df
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
import sys
sys.setrecursionlimit(100000)

import pickle
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'company_df'+'_'+timestamp
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(company_df, to_write)
#%%
filename = 'data/' + 'company_df_2019-12-01-12-23-05.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%