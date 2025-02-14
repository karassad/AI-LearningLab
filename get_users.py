import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_us():


    with open("validate.tsv") as f:
        val = pd.read_csv(f, sep="\t")
    with open("users.tsv") as f:
        user_d = pd.read_csv(f, sep="\t")
    with open("history.tsv") as f:
        hystory = pd.read_csv(f, sep="\t")


    res = pd.DataFrame(columns=['cpm',"start","end","process","size","avg_V","avg_A","avg_CPM"])

    for i in range(0, len(val)): #len(val)
        list_users = val["user_ids"][i].split(',')
        list_users = list(map(int, list_users))

        start = int(val["hour_start"][i])
        end = int(val["hour_end"][i])
        
        #Инфа от users
        sum_V = 0
        sum_A = 0
        sum_CPM = 0
        for j in range(len(list_users)):
            user_info = user_d[user_d["user_id"] == list_users[j]].values[0] #

            sum_A += user_info[2]
            mas = hystory[hystory["user_id"] == user_info[0]]
            
            mas = mas[(mas["hour"] >= start) & (mas["hour"] <= end)]

            c = 0
            for k in range(len(mas)):
                mas_l = list(map(int, mas.iloc[k].values))
                

 
                sum_CPM += mas_l[1]
                c += 1
            sum_V += c
        
        avg_V = round(sum_V/ len(list_users), 2)
        avg_A = round(sum_A / len(list_users), 2)
        if sum_V == 0: 
            avg_CPM = 1000
        else: avg_CPM = round(sum_CPM / sum_V, 2)

        new_row = {'cpm': int(val["cpm"][i]),
                        "start": start,
                        "end" : end,
                        "process" : end - start,
                        "size" : len(list_users),
                        "avg_V" : avg_V,
                        "avg_A" : avg_A,
                        "avg_CPM" : avg_CPM,
                        }
        res.loc[len(res)] = new_row 
        res.iloc[[-1]].to_csv('output2.csv', mode='a', header=False, index=False)

    
    
    res.to_csv('output1.csv', index=False)

             
            


            






    """
    corr_mat = res.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_mat,annot=True)
    plt.title('Correlation Heatmap')
    plt.show()

    for i in res.columns[:-1]:
        plt.hist(res[res['rate']==0][i],color='blue',label='not watch',alpha=0.5)
        plt.hist(res[res['rate']>=1][i],color='red',label='watch 1',alpha=0.5)
        #plt.hist(res[res['rate'] > 1][i],color='green',label='watch > 1',alpha=0.5)

        plt.title(f'Distribution of {i}')
        plt.legend()
        plt.show()
        """


if __name__ == "__main__":
    get_us()