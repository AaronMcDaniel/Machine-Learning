import config
import logging
import numpy



train_file_path = ""
test_file_path = ""
train_data_features = []
train_data_target = []
test_data_features = []
test_data_target = []


"""
Takes 2 args and returns array of length 2
First: first line from csv to put in array
Last: last line from csv to put in array
Element 0: Features data
Element 1: Target data
"""
def getTrainData(first, last, isDota):
  dataNtargets = []
  count = 1

  for line in open(train_file_path, 'r'):
    if count >= first and count <= last:
      split_line = line.rstrip('\n').split(',')
      for i in range(1, len(split_line)):
        if isDota:
          split_line[i] = float(split_line[i])
        else:
          split_line[i] = float(split_line[i])
      train_data_features.append(split_line[1:])
      train_data_target.append(int(split_line[0]))
    elif count > last:
      break
    count += 1

  dataNtargets = [train_data_features, train_data_target]
  return dataNtargets

"""
Takes no args and returns array of length 2
Element 0: Features data
Element 1: Target data
"""
def getTestData(isDota):
  dataNtargets = []

  for line in open(test_file_path, 'r'):
    split_line = line.rstrip('\n').split(',')
    for i in range(1, len(split_line)):
      if isDota:
        split_line[i] = float(split_line[i])
      else:
        split_line[i] = float(split_line[i])
    test_data_features.append(split_line[1:])
    test_data_target.append(int(split_line[0]))

  dataNtargets = [test_data_features, test_data_target]
  return dataNtargets

"""
Specifically returns test and train data for dota in array size 4
Takes 2 args
First: first line from csv to start recording
Last: last line from csv to record
Element 0: Train Feature data
Element 1: Train Target data
Element 2: Test Feature data
element 3: Test Target data
"""
def getDotaData(first, last):
  MAX = 92650

  if last > MAX or first < 1:
    print("ENTER VALID LINES")
    return []

  global train_file_path
  global test_file_path

  train_file_path = config.dota_train_data_path
  test_file_path = config.dota_test_data_path

  train = getTrainData(first, last, True)
  test = getTestData(True)

  allData = [train[0], train[1], test[0], test[1]]
  return allData


"""
Specifically returns test and train data for pulsar (HTRU2) in array size 4
Takes 2 args
First: first line from csv to start recording
Last: last line from csv to record
Element 0: Train Feature data
Element 1: Train Target data
Element 2: Test Feature data
element 3: Test Target data
"""
def getPulsarData(first, last):
  MAX = 10000

  if last > MAX or first < 1:
    print("ENTER VALID LINES")
    return []

  global train_file_path
  global test_file_path

  train_file_path = config.pulsar_train_data_path
  test_file_path = config.pulsar_test_data_path

  train = getTrainData(first, last, False)
  test = getTestData(False)

  allData = [train[0], train[1], test[0], test[1]]
  return allData

  """
  Records data on the reseults of a prediction set. Assumes Binary classification.
  Saves data in csv format in a file.
  Takes 5 args and returns an array of size 7
  INPUTS
  pos: the positive target value
  neg: the negative target value
  outFile: the name of the file the data should be stored in
  targets: correct answers
  test_data_prediction: the outputted answers fomr query
  write: boolean value, of weather to write the results for every instance
  OUTPUTS
  Element1: % of targets correctly predicted
  element2: % of targets that are positive
  Element3: % of targets that are negative
  Element4: % of positive values correctly predicted
  Element5: % of negative values correctly predicted
  Element6:	% of predictions that were false positives
  Element7: % of targets that were false negatives
  """

def recordResults(pos, neg, outFile, targets, test_data_prediction, approach, write):
  correct_count = 0
  variance = 0
  correct_count = 0
  positive_count = 0
  negative_count = 0
  false_positive_count = 0
  false_negative_count = 0
  all_count = len(test_data_prediction)

  fileLines = []
  fileLines.append("Prediction,Target,Variance,,Is correct,False positive,False Negative\n")

  for i in range(all_count):
      falsePositive = 0
      falseNegative = 0

      isCorrect = abs(targets[i] - test_data_prediction[i]) < 0.1
      if isCorrect:
          correct_count += 1
      else:
          variance += 1
          falsePositive = int((int(test_data_prediction[i]) is pos) and (int(targets[i]) is neg))
          falseNegative = int((int(test_data_prediction[i]) is neg) and (int(targets[i]) is pos))

      if targets[i] is pos:
          positive_count += 1
      else:
          negative_count += 1

      line = (str(test_data_prediction[i]) + "," + str(targets[i]) + ","+str(variance)+",,"+str(isCorrect)+","+str(falsePositive)+","+str(falseNegative)+"\n")
      fileLines.append(line)

      false_negative_count += falseNegative
      false_positive_count += falsePositive

  #Add summary information
  per_correct = correct_count * 100.0 / all_count
  per_pos = positive_count * 100.0 / all_count
  per_neg = negative_count * 100.0 / all_count
  per_pos_cor = 100 - (false_negative_count * 100.0 / positive_count)
  per_neg_cor = 100 - (false_positive_count * 100.0 / negative_count)
  fal_pos_per = false_positive_count * 100.0 / all_count
  fal_neg_per = false_negative_count * 100.0 /all_count

  if correct_count is 0:
    write = True

  #includes data on all instances if true
  if(write):
    results = open(approach + outFile, "w")
    results.write("Summary\n")
    results.write("Percent correct," + str(per_correct) + "\n")
    results.write("Percent positive," + str(per_pos) + "\n")
    results.write("Percent negative," + str(per_neg) + "\n")
    results.write("Percent positive correct," + str(per_pos_cor) + "\n")
    results.write("Percent negative correct," + str(per_neg_cor) + "\n")
    results.write("False positive percentage," + str(fal_pos_per) + "\n")
    results.write("False negative percentage," + str(fal_neg_per) + "\n\n")

    for line in fileLines:
      results.write(line)
    results.close()

  logging.info('average variance %d / %d = %f' % (variance, all_count, 1.0 * variance / all_count))
  logging.info('Percent correct: %.5f%%' % (correct_count * 1.0 / all_count * 100))
  logging.info('Percent False correct: %.5f%%' % per_neg_cor)
  logging.info('Percent Positive correct: %.5f%%' % (per_pos_cor))
  return [per_correct, per_pos, per_neg, per_pos_cor, per_neg_cor, fal_pos_per, fal_neg_per]
#end record results

