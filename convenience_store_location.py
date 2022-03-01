#####################################
# 添加包
#####################################
import requests
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import math
import numpy as np

#########################################################
# 将地铁站按线路排序（parameter: （线路名，Dataframe））
#########################################################
def sortStations(line_name, dataframe):
    if 'S1号线/机场线' in line_name:
        dataframe_temp = dataframe.sort_values(by=['latitude'])
        a, b = dataframe_temp.iloc[0], dataframe_temp.iloc[1]
        ab = dataframe_temp.iloc[0].copy()
        dataframe_temp.iloc[0] = b
        dataframe_temp.iloc[1] = ab
        return dataframe_temp
    elif 'S3号线/宁和线' in line_name:
        return dataframe.sort_values(by = ['logitude'])
    elif 'S7号线/宁溧线' in line_name:
        return dataframe.sort_values(by = ['logitude'])
    elif 'S8号线/宁天线' in line_name:
        return dataframe.sort_values(by = ['latitude'])
    elif 'S9号线/宁高线' in line_name:
        return dataframe.sort_values(by = ['latitude'])
    elif '1号线' in line_name:
        df_lines_temp = dataframe.sort_values(by=['latitude'])
        upper_part = df_lines_temp[14:27]
        upper_part = upper_part.iloc[::-1]
        lower_part = df_lines_temp[0:14]
        lower_part = lower_part.sort_values(by=['logitude'])
        a, b = lower_part.iloc[5], lower_part.iloc[6]
        ab = lower_part.iloc[5].copy()
        lower_part.iloc[5] = b
        lower_part.iloc[6] = ab
        df_lines_temp = pd.concat([upper_part, lower_part])
        return df_lines_temp
    elif '2号线' in line_name:
        dataframe['sum'] = dataframe['logitude'] + dataframe['latitude']
        dataframe = dataframe.sort_values(by=['sum'])
        return dataframe
    elif '3号线' in line_name:
        return dataframe.sort_values(by = ['latitude'])
    elif '4号线' in line_name:
        dataframe_temp = dataframe.sort_values(by=['logitude'])
        a, b = dataframe_temp.iloc[16], dataframe_temp.iloc[17]
        ab = dataframe_temp.iloc[16].copy()
        dataframe_temp.iloc[16] = b
        dataframe_temp.iloc[17] = ab
        return dataframe_temp
    elif '10号线' in line_name:
        dataframe_temp = dataframe.sort_values(by=['logitude'])
        a, b, c = dataframe_temp.iloc[8], dataframe_temp.iloc[9], dataframe_temp.iloc[10]
        a_copy = dataframe_temp.iloc[10].copy()
        dataframe_temp.iloc[9] = a
        dataframe_temp.iloc[10] = b
        dataframe_temp.iloc[8] = a_copy
        return dataframe_temp
    else:
        return 'wrong'


#############################
# 使用list 存储信息
#############################
name_list = []      # 便利店名称
logitude_list = []  # 便利店经度
latitude_list = []  # 便利店纬度

# area_code 中包含南京市12个区的adcode
area_code = ['320101','320102','320104','320105','320106','320111','320113','320114','320115','320116','320117','320118']

# 对每一个区域进行检索
for each_area in area_code:
    page_num = 1    # 网页 page number

    #################################################
    # 逐页读取
    # 在高德地图中寻找便利店类型地点时，将types 设为060200 ： types = 060200
    #################################################
    while(True):
        url = 'https://restapi.amap.com/v3/place/text?key= ### &keywords= ### &types=060200&city='+each_area+'&citylimit=true&children=1&offset=25&page='+str(page_num)+'&extensions=all'
        response = requests.get(url)
        dict = response.json()

        list = dict.get('pois')
        if len(list)==0:        # 如果读取不到数据，则跳出循环
            break

        # 提取数据，并放入list中
        for pois in list:
            name_list.append(pois.get('name'))              # 提取便利店店名
            temp_list = pois.get('location').split(',')     # 将经度与纬度分开
            logitude_list.append(float(temp_list[0]))       # 提取经度
            latitude_list.append(float(temp_list[1]))       # 提取纬度

        # 网页翻页
        page_num = page_num+1


################################################
# 将读取的数据，便利店店名+经纬度 存入dataframe
################################################
d = {'name':name_list, 'logitude':logitude_list, 'latitude':latitude_list}
df = pd.DataFrame(d)


########################################
# k-means聚类 + 创建画图所用的dataframe
########################################
number_of_cluster = 300         # 分组组数K
d = {'logitude':logitude_list, 'latitude':latitude_list}
df_draw = pd.DataFrame(d)       # 用来画图的 dataframe
kmeans = KMeans(n_clusters=number_of_cluster,random_state=15).fit(df_draw)  # 实施聚类算法
df_draw['label'] = kmeans.labels_  # 添加画图 datafrane 的序列

# 提取 K-means 所有的聚类中心点
centers = kmeans.cluster_centers_
centers = centers.tolist()
fig, ax = plt.subplots()

# 创建存储数据的list
distance_distribution_list = []     # 每个聚类分组半径
final_radius_list = []              # 符合条件的聚类分组半径
final_logitude_list = []            # 符合筛选条件的分组中心点经度
final_latitude_list = []            # 符合筛选条件的分组中心点纬度
final_data_number_list = []         # 符合筛选条件的分组所含数据个数
number_of_data_list = []            # 每个聚类分组中数据个数

#########################################################
# 对于每一个聚类分组：
# 1. 画出所有此分组内的便利店位置，用同一随机颜色标明
# 2. 如果符合筛选条件，画出聚类分组圆
#########################################################
for i in range(0,number_of_cluster):
    # 根据经纬度画出分组内的便利店位置
    df_temp = df_draw[df_draw['label']==i]  # 提取分组内所有便利店数据

    # 根据经纬度画点
    plt.scatter(df_temp['logitude'], df_temp['latitude'], marker ='.',color=(random.random(),random.random(),random.random()),s=1)

    # 以聚类分组中心点为圆心，计算半径，半径为中心点到此分组内最远便利店的距离
    center = centers[i]                 # 提取中心点数据
    distance_list = []                  # list储存中心点到此分组内每个数据的距离
    x = (df_temp['logitude']).to_list() # 提取此分组内每个便利店的经度数据
    y = (df_temp['latitude']).to_list() # 提取此分组内每个便利店的纬度数据

    # 计算中心点到此分组内每个数据的距离
    for j in range (0,df_temp.iloc[:,0].size):
        distance_list.append(math.hypot(center[0]-x[j], center[1]-y[j]))

    # 提取最远距离作为半径，并存入list中
    distance_distribution_list.append(max(distance_list))

    # 如果此分组符合筛选条件，画出聚类分组圆
    if (max(distance_list)<=0.015 and df_temp.iloc[:,0].size>14):
        # 画圆
        circle = plt.Circle((center[0],center[1]), color='r', radius=max(distance_list), fill=False, linewidth=0.5)
        ax.add_artist(circle)

        # 存储符合筛选条件的分组的相关信息
        final_radius_list.append(max(distance_list))
        final_logitude_list.append(center[0])
        final_latitude_list.append(center[1])
        final_data_number_list.append(df_temp.iloc[:,0].size)

    number_of_data_list.append(df_temp.iloc[:,0].size)      # 存储分组中数据个数信息

plt.show()  # 展示画图

#plt.savefig('final_15_14.png')


##########################
# 根据经纬度得到地址
##########################
current_radius = 0
address_list = []       # 具体地址list
area_list = []          # 区域list

#########################################################
# 对于符合筛选条件的每一个分组：
# 1. 存储分组中心点具体地址
# 2. 存储分组中心点区域
#########################################################
for i in range(0,len(final_logitude_list)):

    url = 'https://restapi.amap.com/v3/geocode/regeo?output=JSON&location='+str(final_logitude_list[i])+','+str(final_latitude_list[i])+'&key= &radius='+str(current_radius)+'&extensions=base'
    response = requests.get(url)
    dict_address = response.json()
    regeocode = dict_address.get('regeocode')

    # 得到并存储具体地址
    address = regeocode.get('formatted_address')
    address = address[6:]   #去掉‘江苏省南京市’
    address_list.append(address)

    # 得到并存储区名
    address = regeocode.get('addressComponent')
    area_list.append(address.get('district'))


# 计算以千米为单位的分组半径
final_radius_list_km = [i*111 for i in final_radius_list]

############
# 输出表格
############
circle_result1 = {'number of data': final_data_number_list, 'logitude': final_logitude_list,
                  'latitude': final_latitude_list, 'radius': final_radius_list}
circle_result2 = {'number of data': final_data_number_list, 'address':address_list,
                  'area': area_list,'radius': final_radius_list_km}
circle_df = pd.DataFrame(circle_result1)
circle_df = circle_df.sort_values(by=['number of data'])    # 将数据以分组内所含数据量由高到低排序
circle_df = circle_df.iloc[::-1]
circle_df.to_excel('circle_result1.xlsx')
circle_df = pd.DataFrame(circle_result2)
circle_df = circle_df.sort_values(by=['number of data'])    # 将数据以分组内所含数据量由高到低排序
circle_df = circle_df.iloc[::-1]
circle_df.to_excel('circle_result2.xlsx')



'''
plt.xlim(118.5,119.1)
plt.ylim(31.6,32.1)
plt.show()
plt.savefig('final.png')
'''

'''
#####################
# 画分组半径分布图
#####################
import matplotlib.pyplot as plt1
n,bins,patches = plt1.hist(distance_distribution_list, bins = 'auto')
plt1.xlim(0,0.1)
plt1.ylim(0,85)
plt1.show()

#######################
# 画分组数据个数分布图
#######################
import matplotlib.pyplot as plt1
n,bins,patches = plt1.hist(number_of_data_list, bins = 'auto')
plt1.xlim(0,300)
plt1.ylim(0,115)
plt1.show()
'''




################
# 添加地铁线路
################
page_num = 1            # 网页当前页数
name_list = []          # 地铁站站名
logitude_list = []      # 地铁站经度
latitude_list = []      # 地铁站纬度
line_list = []          # 地铁线路名

###########################################################################
# 逐页读取
# 在高德地图中寻找地铁站类型地点时，以地铁站为关键字搜索： keywords=地铁站
# 将搜索范围定位南京市： city=320100
###########################################################################
while(True):
    # 读取数据
    url = 'https://restapi.amap.com/v3/place/text?key= &keywords=地铁站&types=&city=320100&citylimit=true&children=1&offset=50&page=' + str(page_num) + '&extensions=all'
    response = requests.get(url)
    dict = response.json()

    list = dict.get('pois')
    if len(list)==0:        # 如果读取不到数据，则跳出循环
        break

    # 筛选并存储读到的数据
    for pois in list:
        address = pois.get('address')
        # 如果为地铁站，存储相关信息
        if address[0].isdigit() or 'S' in address[0]:
            # 存储地铁站名+经纬度
            name_list.append(pois.get('name'))
            temp_list = pois.get('location').split(',')
            logitude_list.append(float(temp_list[0]))
            latitude_list.append(float(temp_list[1]))
            # 由于地铁站可能为换乘车站，可能存在多条线路，以list形式储存
            temp_list = address.split(';')
            for each_item in temp_list:
                # 去除在建地铁站
                if '(' in each_item:
                    temp_list.remove(each_item)
            line_list.append(temp_list)

    page_num = page_num+1   # 翻页

# 创建地铁信息dataframe
d = {'name':name_list, 'logitude':logitude_list, 'latitude':latitude_list, 'line': line_list}
df_lines = pd.DataFrame(d)


######################################
# 画出地铁线路
# all_lines：所有地铁线路名称
# color_list： 所有地铁线路对应的颜色
######################################
all_lines = ['S1号线/机场线','S3号线/宁和线','S7号线/宁溧线','S8号线/宁天线','S9号线/宁高线','1号线','2号线','3号线','4号线','10号线']
color_list = [(49/255,86/255,47/255),(218/255,122/255,214/255),(219/255,112/255,147/255),(210/255,105/255,30/255),(225/255,215/255,0),
              (129/255,192/255,219/255),(1,0,0),(50/255,205/255,50/255),(138/255,43/255,226/255),(210/255,180/255,140/255)]

##################################
# 对于每一条地铁线：
# 1. 标出所有地铁站
# 1. 画出地铁线
##################################
for i in range(0,len(all_lines)):
    # 找出所有此地铁线上的地铁站
    mask = df_lines.line.apply(lambda z: all_lines[i] in z)
    df_lines_temp = df_lines[mask]
    df_lines_temp = sortStations(all_lines[i], df_lines_temp)                                       # 将地铁站排序
    plt.scatter(df_lines_temp['logitude'], df_lines_temp['latitude'], marker ='.',color='k',s=15)   # 标出所有地铁站
    plt.plot(df_lines_temp['logitude'], df_lines_temp['latitude'],color = color_list[i])            # 画出地铁线


# 展示画图
plt.xlim(118.5,119.1)
plt.ylim(31.3,32.5)
plt.show()
#plt.savefig('circle and subway.png')
