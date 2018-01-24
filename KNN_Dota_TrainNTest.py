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
dota_size = 92650
dota_step = 92650/100
k = 100

#True will run on various training sets, false will do once
multipleTests = True

dota_time = 0
dota_data = []
test_data_prediction = []


nbrs = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
headers = "Neighbors,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"


if multipleTests:
    file = open("./KNN/KNN_" + config.dota_analysis, 'w')
    file.write(headers)

    for i in range(100): # for testing
      trialTime = datetime.datetime.now()
      nbrs = KNeighborsClassifier(n_neighbors=(i + 1), algorithm='kd_tree')
      line = "%d," % ((i + 1) * dota_step)

      logging.info('Getting Data............')
      dota_data = data_handler.getDotaData(1, (i + 1) * dota_step)

      logging.info('Fitting Data............')
      nbrs.fit(dota_data[0], dota_data[1])

      logging.info('Testing Data............')
      test_data_prediction = nbrs.predict(dota_data[2])

      res = data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, "./KNN/KNN_", False)

      trialTime = (datetime.datetime.now() - trialTime).seconds
      line += "%.2f," %trialTime

      for item in res:
        line += "%.5f%%," % item
      line = line[:-1] + "\n"
      file.write(line)
      logging.info("FINISHED TEST #%d/100 time: %d" % ((i + 1), trialTime))


    file.close()

else:
    dota_data = data_handler.getDotaData(1, dota_size)
    nbrs.fit(dota_data[0], dota_data[1])
    test_data_prediction = nbrs.predict(dota_data[2])
    data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, "./KNN/KNN_", True)

end_time = datetime.datetime.now()
dota_time = (end_time - start_time).seconds
#End dota

logging.info('total dota running time: %.2f seconds' % dota_time)
logging.info('end program-----------------')
