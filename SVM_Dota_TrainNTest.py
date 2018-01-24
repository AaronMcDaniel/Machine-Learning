import config
import data_handler
import logging
import datetime
from sklearn.svm import SVC

start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')

# info for dota
dota_size = 92650
dota_step = 92650/10

#True will run on various training sets, false will do once
multipleTests = True
rbf = "rbf"
linear = "linear"
poly = "poly"
sigmoid = "sigmoid"
precomputed = "precomputed"

kern = linear

approach = "./SVM/SVM_"
dota_time = 0
dota_data = []
test_data_prediction = []

svm = SVC(kernel = kern)
headers = "Size,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"

if multipleTests:
    file = open(approach + config.dota_analysis, 'w')
    file.write(kern + "\n")
    file.write(headers)

    for i in range(3, 5): # for testing
      trialTime = datetime.datetime.now()
      svm = SVC(kernel = kern)
      line = "%d," % ((i + 1) * dota_step)

      logging.info('Getting Data............')
      dota_data = data_handler.getDotaData(1, (i + 1) * dota_step)

      logging.info('Fitting Data............')
      svm.fit(dota_data[0], dota_data[1])

      logging.info('Testing Data............')
      test_data_prediction = svm.predict(dota_data[2])

      res = data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, approach, False)

      trialTime = (datetime.datetime.now() - trialTime).seconds
      line += "%.2f," %trialTime

      for item in res:
        line += "%.5f%%," % item
      line = line[:-1] + "\n"
      file.write(line)
      logging.info("FINISHED TEST #%d/10 time: %d" % ((i + 1), trialTime))

    file.close()

else:
    logging.info('getting data.............')
    dota_data = data_handler.getDotaData(1, dota_size)
    logging.info('fitting data.............')
    svm.fit(dota_data[0], dota_data[1])
    logging.info('predicting data..........')
    test_data_prediction = svm.predict(dota_data[2])
    data_handler.recordResults(1, -1, config.dota_results,dota_data[3], test_data_prediction, approach, True)

end_time = datetime.datetime.now()
dota_time = (end_time - start_time).seconds
#End dota

logging.info('total dota running time: %.2f seconds' % dota_time)
logging.info('end program-----------------')
