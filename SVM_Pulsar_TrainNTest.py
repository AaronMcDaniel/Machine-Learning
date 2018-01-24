"""
Will end up in main directory probably
Applies the KNN approach to Pulsar dataset
"""

import config
import data_handler
import logging
import datetime
from sklearn.svm import SVC

start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')


# info for data
pulsar_size = 10000
pulsar_step = 10000/10
#True will run on various training sets, false will do once
multipleTests = True

rbf = "rbf"
linear = "linear"
poly = "poly"
sigmoid = "sigmoid"
precomputed = "precomputed"

kern = rbf
approach = "./SVM/SVM_"
pulsar_time = 0
pulsar_data =[]
test_data_prediction = []


###APPLY KNN TO PULSAR DATA###
svm = SVC(kernel = kern)
headers = "Size,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"

if multipleTests:
    file = open(approach + config.pulsar_analysis, 'w')
    file.write(kern + "\n")
    file.write(headers)
    err_cnt = 0

    for i in range(3):
      trialTime = datetime.datetime.now()
      svm = SVC(kernel = kern)
      line = "%d," % ((i + 1) * pulsar_step)

      logging.info('Getting Data............')
      pulsar_data = data_handler.getPulsarData(1, (i + 1) * pulsar_step)

      logging.info('Fitting Data............')
      svm.fit(pulsar_data[0], pulsar_data[1])

      logging.info('Testing Data............')
      test_data_prediction = svm.predict(pulsar_data[2])

      res = data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, approach, False)

      trialTime = (datetime.datetime.now() - trialTime).seconds
      line += "%.2f," %trialTime

      for item in res:
        line += '%.5f,' % item
      line = line[:-1] + "\n"
      file.write(line)

      logging.info("FINISHED TEST #%d/10 time: %d" % ((i + 1), trialTime))

    file.close()
else:
    pulsar_data = data_handler.getPulsarData(1, pulsar_size)
    svm.fit(pulsar_data[0], pulsar_data[1])
    test_data_prediction = svm.predict(pulsar_data[2])
    data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, approach, True)

end_time = datetime.datetime.now()
pulsar_time = (end_time - start_time).seconds
start_time = datetime.datetime.now()
# End pulsar
##############################

logging.info('total pulsar running time: %.2f seconds' % pulsar_time)
logging.info('end program-----------------')
