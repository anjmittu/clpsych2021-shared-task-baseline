import argparse
import json
import statistics

def read_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for json_obj in f:
            data.append(json.loads(json_obj))

    return data

def main(data_file):
    data = read_json_data(data_file)

    print("Number of users: {}".format(len(data)))
    print(" ")

    print("Number of users with attempts: {}".format(len([user for user in data if user["has_attempt"] == 1])))
    print("Number of control users: {}".format(len([user for user in data if user["has_attempt"] == 0])))
    print(" " )
 
    num_tweets = [len(user["tweets"]) for user in data]
    print("Number of tweets per each user")
    print("Average: {}".format(sum(num_tweets) // len(num_tweets)))
    print("Median: {}".format(statistics.median(num_tweets)))
    print("Max: {}".format(max(num_tweets)))
    print("Min: {}".format(min(num_tweets)))
    print(" ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prints stats about a data file')
    parser.add_argument('--data_file', help='The data file which stats will be printed for', type=str)
    args = parser.parse_args()
    main(args.data_file)
