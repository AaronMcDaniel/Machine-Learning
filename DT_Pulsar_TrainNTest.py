

import config
import data_handler
import logging
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')


# info for data
pulsar_size = 10000
pulsar_step = 10000/100
best_depth = 3
best_leaf = 10
best_split = 100

#True will run on various training sets, false will do once
multipleTests = False

pulsar_time = 0
pulsar_data =[]
test_data_prediction = []


clf = DecisionTreeClassifier(max_depth = best_depth, min_samples_leaf = best_leaf, min_samples_split = best_split)
headers = "Min split,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"

if multipleTests:
    file = open("./DT/_DT_" + config.pulsar_analysis, 'w')
    file.write("leaf,split = 100,depth = 9\n")
    file.write(headers)

    for i in range(1,6):
      trialTime = datetime.datetime.now()
      clf = DecisionTreeClassifier(max_depth = best_depth, min_samples_leaf = best_leaf, min_samples_split = pow(10,i))
      line = "%d," % (pow(10,i) )

      logging.info('Getting Data............')
      pulsar_data = data_handler.getPulsarData(1, pulsar_size)

      logging.info('Fitting Data............')
      clf.fit(pulsar_data[0], pulsar_data[1])

      logging.info('Testing Data............')
      test_data_prediction = clf.predict(pulsar_data[2])

      res = data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, "./DT/DT_", False)

      trialTime = (datetime.datetime.now() - trialTime).seconds
      line += "%.2f," %trialTime

      for item in res:
        line += '%.5f,' % item
      line = line[:-1] + "\n"
      file.write(line)

      logging.info("FINISHED TEST #%d/20 time: %d" % ((i + 1), trialTime))

    file.close()
else:
    pulsar_data = data_handler.getPulsarData(1, pulsar_size)
    clf.fit(pulsar_data[0], pulsar_data[1])
    test_data_prediction = clf.predict(pulsar_data[2])
    logging.info(str(len(test_data_prediction)))
    data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction, "./DT/DT_", True)

#print(classification_report(pulsar_data[3], test_data_prediction))

end_time = datetime.datetime.now()
pulsar_time = (end_time - start_time).seconds
start_time = datetime.datetime.now()
# End pulsar
##############################

logging.info('total pulsar running time: %.2f seconds' % pulsar_time)
logging.info('end program-----------------')
