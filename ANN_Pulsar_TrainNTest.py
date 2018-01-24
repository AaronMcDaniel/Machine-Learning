#!/usr/bin/python
#encoding=utf8

import config
import data_handler
import logging
import datetime
from sklearn.neural_network import MLPClassifier

start_time = datetime.datetime.now()
FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('start program-----------------')


# info for data
pulsar_size = 10000
pulsar_step = 10000/10
multipleTests = True
learn = 0.01

pulsar_time = 0
pulsar_data =[]
test_data_prediction = []


#clf = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes(50,13), random_state=1)
clf = MLPClassifier(solver='lbfgs',alpha = 1e-5,hidden_layer_sizes = (50,13), random_state=1)
headers = "size,Time (s),Percent Correct, Percent +, Percent -, Percent + correct, Percent - correct, False + percent, False - percent\n"

if multipleTests:
    file = open("./ANN/ANN_" + config.pulsar_analysis, 'w')
    file.write(str(learn) + "\n")
    file.write(headers)

    for i in range(10):
        trialTime = datetime.datetime.now()
        wide = 60
        deep = 10
        clf = MLPClassifier(solver='lbfgs', learning_rate_init = learn ,hidden_layer_sizes = (wide,deep), random_state=1)
        line = "%d," % ((i + 1) * pulsar_step)

        logging.info('Getting Data............')
        pulsar_data = data_handler.getPulsarData(1, (i + 1) * pulsar_step)

        logging.info('Fitting Data............')
        clf.fit(pulsar_data[0], pulsar_data[1])

        logging.info('Testing Data............')
        test_data_prediction = clf.predict(pulsar_data[2])

        res = data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3],test_data_prediction,"./ANN/ANN_",False)

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
    clf.fit(pulsar_data[0], pulsar_data[1])
    test_data_prediction = clf.predict(pulsar_data[2])
    data_handler.recordResults(1, 0, config.pulsar_results, pulsar_data[3], test_data_prediction, "./ANN/ANN_", True)

end_time = datetime.datetime.now()
pulsar_time = (end_time - start_time).seconds
start_time = datetime.datetime.now()
# End pulsar
##############################

logging.info('total pulsar running time: %.2f seconds' % pulsar_time)
logging.info('end program-----------------')
