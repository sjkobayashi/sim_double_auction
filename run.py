from model import *
import pandas as pd

# ------------------------------ Simulation ------------------------------

# ZIP
#supply = Supply(6, 5, 1, 75, 200, 1)
#demand = Demand(6, 5, 1, 325, 200, 1)

# GD
#supply = Supply(6, 5, 1, 1.45, 2.50, 1)
#demand = Demand(6, 5, 1, 3.55, 2.50, 1)

# test


def batcher(supply, demand, s_strategy, b_strategy, max_steps,
            highest_ask, lowest_bid):
    print(s_strategy, b_strategy)
    datas = list()

    batch_index = 1
    while batch_index <= 100:
        print("Batch:", batch_index)
        model = CDAmodel(supply, demand, s_strategy, b_strategy,
                         highest_ask, lowest_bid)
        for j in range(10):
            print("Period", model.num_period)
            for i in range(max_steps):
                model.step()
            model.next_period()
            if model.no_trade:
                print("No Trade")
                break
        else:
            data_model = model.datacollector.get_model_vars_dataframe()
            data_t_model = data_model[data_model.Traded == 1].drop(
                columns='Traded')
            data_t_model['batch'] = batch_index
            datas.append(data_t_model)
            batch_index += 1

    batch_data = pd.concat(datas)
    cols = batch_data.columns.tolist()
    cols = cols[-1:] + cols[6:4:-1] + cols[8:9] + cols[:5] + cols[7:8]
    batch_data = batch_data[cols]
    return batch_data

# Model 1
supply = Supply(5, 5, 1, 14.50, 25.00, 1)
demand = Demand(5, 5, 1, 35.50, 25.00, 1)

ZI_batch = batcher(supply, demand, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv("simulation_results/ZI_1.csv", index=False)

ZIP_batch = batcher(supply, demand, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv("simulation_results/ZIP_1.csv", index=False)

GD_batch = batcher(supply, demand, MGD, MGD, 500, 100, 0)
GD_batch.to_csv("simulation_results/MGD_1.csv", index=False)


# Model 2
supply = Supply(5, 5, 1, 24.00, 25.00, 1)
demand = Demand(5, 5, 1, 35.50, 25.00, 1)

ZI_batch = batcher(supply, demand, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv("simulation_results/ZI_2.csv", index=False)

ZIP_batch = batcher(supply, demand, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv("simulation_results/ZIP_2.csv", index=False)

GD_batch = batcher(supply, demand, MGD, MGD, 500, 100, 0)
GD_batch.to_csv("simulation_results/MGD_2.csv", index=False)


# Model 3
supply = Supply(5, 5, 1, 14.50, 25.00, 1)
demand = Demand(5, 5, 1, 30.00, 25.00, 1)

ZI_batch = batcher(supply, demand, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv("simulation_results/ZI_3.csv", index=False)

ZIP_batch = batcher(supply, demand, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv("simulation_results/ZIP_3.csv", index=False)

GD_batch = batcher(supply, demand, MGD, MGD, 500, 100, 0)
GD_batch.to_csv("simulation_results/MGD_3.csv", index=False)
