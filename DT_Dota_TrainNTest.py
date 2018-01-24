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
dota_size = 92650
dota_step = 92650/100

#True will run on various training sets, false will do once
multipleTests = True

dota_time = 0
dota_data = []
test_data_prediction = []
best_depth = 6
best_leaf = 6
best_split = 32


clf = DecisionTreeClassifier()
headers = "Leaf,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"


if multipleTests:
    file = open("./DT/Split_DT_" + config.dota_analysis, 'w')
    file.write("depth,leaf = 6,split = 32\n")
    file.write(headers)

    for i in range(20): # for testing
      trialTime = datetime.datetime.now()
      clf = DecisionTreeClassifier(max_depth = (i +1)*2, min_samples_leaf = (i +1) * 2, min_samples_split = best_split)
      line = "%d," % ((i + 1) * 2)

      logging.info('Getting Data............')
      dota_data = data_handler.getDotaData(1, (i + 1) * dota_step)

      logging.info('Fitting Data............')
      clf.fit(dota_data[0], dota_data[1])

      logging.info('Testing Data............')
      test_data_prediction = clf.predict(dota_data[2])

      res = data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, "./DT/DT_", False)

      trialTime = (datetime.datetime.now() - trialTime).seconds
      line += "%.2f," %trialTime

      for item in res:
        line += "%.5f%%," % item
      line = line[:-1] + "\n"
      file.write(line)
      logging.info("FINISHED TEST #%d/20 time: %d" % ((i + 1), trialTime))


    file.close()

else:
    dota_data = data_handler.getDotaData(1, dota_size)
    clf.fit(dota_data[0], dota_data[1])
    test_data_prediction = clf.predict(dota_data[2])
    data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, "./DT/DT_", True)

end_time = datetime.datetime.now()
dota_time = (end_time - start_time).seconds
#End dota

logging.info('total dota running time: %.2f seconds' % dota_time)
logging.info('end program-----------------')
