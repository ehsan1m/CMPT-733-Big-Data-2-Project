import os
import pandas as pd


input_dir = "./FB_Scraping/cnn"
ouput_file = "./FB_Scraping/cnn_data.csv"

df_list = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        df = pd.DataFrame.from_csv(root + "/" + file)
        df_list.append(df)


final_df = pd.concat(df_list, ignore_index=True)
final_df.sample(n=200000, random_state=10).to_csv(ouput_file, index=False)
#final_df.to_csv(ouput_file, index=False)