from model import *
import pandas as pd

# ------------------------------ Graph --------------------------
plt.step(np.append(0, supply.cumulative_quantity),
         np.append(supply.price_schedule[0], supply.price_schedule),
         label="Supply", color="blue")
plt.step(np.append(0, supply2.cumulative_quantity),
         np.append(supply2.price_schedule[0], supply2.price_schedule),
         label="Supply", color="darkcyan")
plt.step(np.append(0, demand.cumulative_quantity),
         np.append(demand.price_schedule[0], demand.price_schedule),
         label="Demand", color="red")
plt.step(np.append(0, demand2.cumulative_quantity),
         np.append(demand2.price_schedule[0], demand2.price_schedule),
         label="Demand", color="darkred")
axes = plt.gca()
plt.vlines(supply.num_in, 0, supply.equilibrium_price,
           linestyle="dashed")
plt.hlines(demand.equilibrium_price, 0, demand.num_in,
           linestyle="dashed")
axes.set_xlabel("Quantity")
axes.set_ylabel("Price")
plt.legend()
plt.show()


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
    while batch_index <= 500:
        print("Batch:", batch_index)
        model = CDAmodel(supply, demand, s_strategy, b_strategy,
                         highest_ask, lowest_bid)
        for j in range(10):
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
supply1 = Supply(6, 5, 1, 15.00, 25.00, 1)
demand1 = Demand(6, 5, 1, 35.00, 25.00, 1)

# Model 2
supply2 = Supply(6, 5, 1, 24.00, 25.00, 1)
demand2 = Demand(6, 5, 1, 35.00, 25.00, 1)

# Model 3
supply3 = Supply(6, 5, 1, 15.00, 25.00, 1)
demand3 = Demand(6, 5, 1, 26.00, 25.00, 1)

# Model 4
supply4 = Supply(6, 5, 1, 20.00, 25.00, 1)
demand4 = Demand(6, 5, 1, 30.00, 25.00, 1)

#supply4.market_graph(demand4, ylim=[-2, 37])

file_path = "simulation_results/batch_1/"

# ZI
print("\n ZI \n")

ZI_batch = batcher(supply1, demand1, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv(file_path + "ZI_1.csv", index=False)

ZI_batch = batcher(supply2, demand2, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv(file_path + "ZI_2.csv", index=False)

ZI_batch = batcher(supply3, demand3, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv(file_path + "ZI_3.csv", index=False)

ZI_batch = batcher(supply4, demand4, ZI, ZI, 500, 100, 0)
ZI_batch.to_csv(file_path + "ZI_4.csv", index=False)


# ZIP
print("\n ZIP \n")

ZIP_batch = batcher(supply1, demand1, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv(file_path + "ZIP_1.csv", index=False)

ZIP_batch = batcher(supply2, demand2, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv(file_path + "ZIP_2.csv", index=False)

ZIP_batch = batcher(supply3, demand3, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv(file_path + "ZIP_3.csv", index=False)

ZIP_batch = batcher(supply4, demand4, ZIP, ZIP, 500, 100, 0)
ZIP_batch.to_csv(file_path + "ZIP_4.csv", index=False)


# GD
print("\n GD \n")

GD_batch = batcher(supply1, demand1, MGD, MGD, 500, 100, 0)
GD_batch.to_csv(file_path + "MGD_1.csv", index=False)

GD_batch = batcher(supply2, demand2, MGD, MGD, 500, 100, 0)
GD_batch.to_csv(file_path + "MGD_2.csv", index=False)

GD_batch = batcher(supply3, demand3, MGD, MGD, 500, 100, 0)
GD_batch.to_csv(file_path + "MGD_3.csv", index=False)

GD_batch = batcher(supply4, demand4, MGD, MGD, 500, 100, 0)
GD_batch.to_csv(file_path + "MGD_4.csv", index=False)


