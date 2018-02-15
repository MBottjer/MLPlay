def gini_index(targets):
	""" 
	Purpose: this function determines the 'impurity' of a split
			 by providing a value between 0 and 1 where 0 indi-
			 cates a perfect split.
	"""
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
			# calculate the sum of the squared class probabilities
			class_prob_squared = sum([(float(split_i.count(target))/observations_in_split)**2 for target in possible_targets])
			# sum weighted gini index
			gini_index += (1.0 - class_prob_squared) * relative_size
	return gini_index

def split_data(i,given_value,data,targets):
	""" 
	Purpose: split the data to the right if the relevant record
			 is the same as the given value. If not, append to
			 the left branch.
	"""
	left_branch, right_branch, left_targets, right_targets = ([] for i in xrange(4))
	for record,target in zip(data, targets):
		if record[i] != given_value:
			left_branch.append(record)
			left_targets.append(target)
		else:
			right_branch.append(record)
			right_targets.append(target)
	return left_branch, right_branch, left_targets, right_targets

def get_optimal_split(data, targets):
	""" 
	Purpose: find the optimal split by determining the minimum gini
			 index for each proposed split.
	"""
	# start with an arbitrarily large gini score
	best_gini_index = 1
	# we will search through each explanatory variable
	# for each record in the data
	for i in xrange(len(data[0])):
		for record in data:
			potential_split = split_data(i, record[i], data, targets)
			# the potential split contains a potential left and right branch
			# for both the data and targets. To calculate the gini impurity
			# we need the targets in the left and right branch i.e. the last
			# two arrays.
			gini_impurity = gini_index(potential_split[2:4])
			# update the gini impurity if the proposed split is an improvement
			if gini_impurity < best_gini_index:
				index_of_split, split_value, best_gini_index, best_split = i, record[i], gini_impurity, potential_split
	return {'index':index_of_split, 'value':split_value, 'split':best_split}

def find_mode(targets):
	""" 
	Purpose: find the most common target in an array of targets
	"""
	max_frequency = max([targets.count(target) for target in targets])
	if max_frequency > 1:
		return [value for value in targets if targets.count(value) == max_frequency][0]
	else:
		# if we have an equal number of target values or just one then
		# we pick the first value in targets.
		return targets[0]

def split(node, max_depth, min_size, depth):
	""" 
	Purpose: given an initial split, node, recursively create
			 branches, taking both the maximum depth and
			 minimum number of observations into account.
	To Note: there are two conditions to stop the recursive
			 process, the first is checking no further splits
			 can be made and the second, we don't breach our
			 maximum depth.
	"""
	left_data, right_data,left_target, right_target = node['split']
	del(node['split'])
	# at end nodes one of the two branches will be empty and hence,
	# they should predict equal outcomes.
	if not left_data or not right_data:
		node['left'] = node['right'] = find_mode(left_target + right_target)
		return node
	# if the tree has reached the maximum depth then we assign the
	# left and right nodes their respective predicted outcomes.
	if depth >= max_depth:
		node['left'], node['right'] = find_mode(left_target), find_mode(right_target)
		return node

	# in the event that a split resulted in a child node with
	# equal or fewer records than the minimum sample then do
	# not split further. Furthermore, if another split results in the 
	# unecessary branches then do not split. Otherwise, split further.
	if len(left_target) <= min_size or len(left_target) == sum(left_target):
		node['left'] = find_mode(left_target)
	else:
		node['left'] = get_optimal_split(left_data, left_target)
		split(node['left'], max_depth, min_size, depth+1)

	if len(right_target) <= min_size or len(right_target) == sum(right_target):
		node['right'] = find_mode(right_target)
	else:
		node['right'] = get_optimal_split(right_data, right_target)
		split(node['right'], max_depth, min_size, depth+1)

def fit(X_training,y_training, max_depth, min_size):
	""" 
	Purpose: construct a tree to later be used for prediction
	"""
	root_node = get_optimal_split(X_training,y_training)
	split(root_node, max_depth, min_size, 1)
	return root_node

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[Var %d = %d]' % ((depth*'\t', (node['index']+1), node['value'])))
		depth+=1
		print_tree(node['left'], depth)
		print_tree(node['right'], depth)
	else:
		# print predicted class in end node
		print('%s[%s]' % ((depth*'\t', node)))

def predict(node, sample):
	if sample[node['index']] == node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], sample)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], sample)
		else:
			return node['right']