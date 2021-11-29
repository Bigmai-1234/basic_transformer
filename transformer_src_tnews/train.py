from data_prepare_util import Dataset
import tensorflow as tf
from metrics import get_binary_metrics, get_multi_metrics
from transformer_code import Transformer
from utils import Config, fixedPositionEmbedding, nextBatch
import os
import datetime
from metrics import mean

config = Config()
data = Dataset(config)
data.dataGen()
# 训练模型
# 生成训练集和验证集
trainReviews = data.trainX
trainLabels = data.trainY
evalReviews = data.evalX
evalLabels = data.evalY
labelList = list(range(len(data.labels)))
embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength)

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        transformer = Transformer(config)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(transformer.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", transformer.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        savedModelPath = config.savedModelPath
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            batchY = batchY.astype(int)
            feed_dict = {
                transformer.inputX: batchX,
                transformer.inputY: batchY,
                transformer.dropoutKeepProb: config.model.dropoutKeepProb,
                transformer.embeddedPosition: embeddedPosition
            }
            _, summary, step, loss, predictions,l3loss = sess.run(
                [trainOp, summaryOp, globalStep, transformer.loss, transformer.predictions,transformer.l3Loss],
                feed_dict)

            print(l3loss)
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

            elif config.numClasses > 1:
                import pdb;pdb.set_trace()
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,labels=labelList)

            trainSummaryWriter.add_summary(summary, step)

            return round(loss,4), round(acc, 4), round(prec, 4), round(recall, 4), round(f_beta, 4)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            batchY = batchY.astype(int)
            feed_dict = {
                transformer.inputX: batchX,
                transformer.inputY: batchY,
                transformer.dropoutKeepProb: 1.0,
                transformer.embeddedPosition: embeddedPosition
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, transformer.loss, transformer.predictions],
                feed_dict)
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)
            trainSummaryWriter.add_summary(summary, step)
            return round(loss,4), round(acc, 4), round(prec, 4), round(recall, 4), round(f_beta, 4)

        best_acc = 0
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model...........")
            for bi, batchTrain in enumerate(nextBatch(trainReviews, trainLabels, config.batchSize)):
                # import pdb;pdb.set_trace()
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                if bi % 20 == 0:
                    print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    # import pdb;pdb.set_trace()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(
                                                                                                             precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))

                if currentStep % config.training.checkpointEvery == 0 and mean(accs) > best_acc:
                    best_acc = mean(accs)
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, config.ckptPath, global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()