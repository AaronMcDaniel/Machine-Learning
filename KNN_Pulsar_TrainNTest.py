"""
Will end up in main directory probably
Applies the KNN approach to Pulsar dataset
"""

import config
import data_handler
import logging
import datetime
from sklearn.neighbors import KNeighborsClassifier

start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')


# info for data
pulsar_size = 10000
pulsar_step = 10000/100
#True will run on various training sets, false will do once
multipleTests = True
k = 10

pulsar_time = 0
pulsar_data =[]
test_data_prediction = []


###APPLY KNN TO PULSAR DATA###
nbrs = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
headers = "Neighbors,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"

if multipleTests:
    file = open("./KNN/testKNN_" + config.pulsar_analysis, 'w')
    file.write(headers)

    for i in range(4):
      trialTime = datetime.datetime.now()
      nbrs = KNeighborsClassifier(n_neighbors=(pow(10,i)), algorithm='kd_tree')
      line = "%d," % ((i + 1) * pulsar_step)

      logging.info('Getting Data............')
      pulsar_data = data_handler.getPulsarData(1, pulsar_size)

      logging.info('Fitting Data............')
      nbrs.fit(pulsar_data[0], pulsar_data[1])

      logging.info('Testing Data............')
      test_data_prediction = nbrs.predict(pulsar_data[2])

      res = data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, "./KNN/KNN_", False)

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
    nbrs.fit(pulsar_data[0], pulsar_data[1])
    test_data_prediction = nbrs.predict(pulsar_data[2])
    data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, "./KNN/KNN_", True)

end_time = datetime.datetime.now()
pulsar_time = (end_time - start_time).seconds
start_time = datetime.datetime.now()
# End pulsar
##############################

logging.info('total pulsar running time: %.2f seconds' % pulsar_time)
logging.info('end program-----------------')
