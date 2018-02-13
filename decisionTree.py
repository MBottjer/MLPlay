def gini_index(targets):
	gini_index = 0.0
	# number of observations across both samples.
	total_number_of_observations = len(sum(targets,[]))
	# classes the targets can take on
	possible_targets = list(set(sum(targets,[])))
	# We iterate through the two samples and determine how well divided they are
	for split_i in targets:
		observations_in_split = len(split_i)
		# at end nodes we will not need to split further
		if observations_in_split is not 0:
			relative_size = float(observations_in_split)/total_number_of_observations
			score = sum([(float(split_i.count(target))/observations_in_split)**2 for target in possible_targets])
			# calculate weighted gini index
			gini_index += (1.0 - score) * relative_size
	return gini_index

def split_data(i,given_value,data, targets):
	left_branch, right_branch = [], []
	left_targets, right_targets = [], []
	for sample,target in zip(data, targets):
		if sample[i] == given_value:
			left_branch.append(sample)
			left_targets.append(target)
		else:
			right_branch.append(sample)
			right_targets.append(target)
	return left_branch, right_branch, left_targets, right_targets

def get_optimal_split(data, targets):
	best_gini_score = 100
	for explanatory_variable_i in xrange(len(data[0])):
		for sample in data:
			potential_split = split_data(explanatory_variable_i, sample[explanatory_variable_i], data, targets)
			gini_score = gini_index(potential_split[2:4])
			if gini_score < best_gini_score:
				index_of_split, split_value, best_gini_score, best_split = explanatory_variable_i, sample[explanatory_variable_i], gini_score, potential_split
				print(gini_score)
	return {'index':index_of_split, 'value':split_value, 'split':best_split}

def most_present_class(targets):
	return max(set(targets), key=targets.count)

def split(node, max_depth, min_size, depth):
	left_data, right_data,left_target, right_target = node['split']
	del(node['split'])
	# check for a no split
	if not left_data or not right_data:
		node['left'] = node['right'] = most_present_class(left_target + right_target)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = most_present_class(left_target), most_present_class(right_target)
		return
	# process left child
	if len(left_target) <= min_size:
		node['left'] = most_present_class(left_target)
	else:
		node['left'] = get_optimal_split(left_data, left_target)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right_target) <= min_size:
		node['right'] = most_present_class(right_target)
	else:
		node['right'] = get_optimal_split(right_data, right_target)
		split(node['right'], max_depth, min_size, depth+1)

def decision_tree(X_training,y_training, max_depth, min_size):
	root_node = get_optimal_split(X_training,y_training)
	split(root_node, max_depth, min_size, 1)
	return root_node