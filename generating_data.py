import random

# generate random multiplication results and dump them in csv file
 
def data_gen(num_data_points):

    real_num_arrays = [None]*num_data_points

    for i in range(num_data_points):
        real_num_arrays[i]=random.uniform(0.01,10000)

    random.shuffle(real_num_arrays)

    return real_num_arrays

def dataset(total_data_points):
    half_data_points = total_data_points//2

    total_space = data_gen(total_data_points)
    x1 = total_space[:half_data_points]
    x2 = total_space[half_data_points:]

    y = [None]*(half_data_points)

    for i in range(half_data_points):
        y[i] = x1[i]*x2[i]

    return (x1,x2,y)

if __name__ == "__main__":
    
    total_data_points = 100000
    
    x1,x2,y = dataset(total_data_points)
    
    f = open("data.csv","a")

    for i in range(len(x1)):
        f.write(str(x1[i])+","+str(x2[i])+","+str(y[i]))
        f.write("\n")
    
    f.close()

    print("\nData generated in data.csv file.\n")