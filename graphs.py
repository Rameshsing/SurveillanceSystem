import matplotlib.pyplot as plt

def plot_traffic_logs(log_data):
    timestamps = [log["time"] for log in log_data]
    counts_in = [log["in"] for log in log_data]
    counts_out = [log["out"] for log in log_data]

    plt.plot(timestamps, counts_in, label="In")
    plt.plot(timestamps, counts_out, label="Out")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Traffic Over Time")
    plt.legend()
    plt.show()
