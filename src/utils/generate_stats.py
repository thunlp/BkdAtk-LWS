def generate_rule_based_replacement_stats(stats_raw):
    poisoned_nums = []
    for i in stats_raw:
        poisoned_nums.append(i['poisoned_num'])
    print("poisoned_nums: ", poisoned_nums)
    print("avg: ", sum(poisoned_nums)/len(poisoned_nums))