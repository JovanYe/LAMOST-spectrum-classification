import torch
import numpy as np
import torch.nn as nn
from os import listdir
from lamostModel import Model
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from dataProcessing_and_visualization import  sample_expansion


def eval(GALAXY_flux, QSO_flux, STAR_flux, epochs):
    sample_size = 3500
    batch_num = 0
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Loss = []
    batch = []
    acc = []
    rec = []
    epoch_line = []
    class_names = ['GALAXY', 'QSO', 'STAR']
    model.to(device='cuda')
    for epoch in range(epochs):
        indices_to_duplicate = np.random.choice(range(len(STAR_flux)), size=sample_size, replace=True)  # 重采样
        STAR_flux_ = STAR_flux[indices_to_duplicate]

        data = torch.from_numpy(np.vstack((GALAXY_flux, QSO_flux, STAR_flux_))).to(device=device).float()
        label = torch.from_numpy(np.concatenate([np.full(sample_size, 0), np.full(sample_size, 1), np.full(sample_size, 2)], axis=0)).to(device=device).float()

        data = data.unsqueeze(1)  # torch.Size([10500, 1, 343])
        print(data.shape, label.shape)
        x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.1, shuffle=True)


        train_dataset = Data.TensorDataset(x_train, y_train.long())
        test_dataset = Data.TensorDataset(x_test, y_test.long())
        print("#####################################epoch:" + str(epoch + 1))
        ##    train carbon
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=32,
                                             shuffle=True,
                                             drop_last=True)  # 丢弃最后一个batch, 避免模型震荡

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=16,
                                                   shuffle=True,
                                                   drop_last=False)
        model.train()
        epoch_line.append(batch_num)

        for i, (flux, labels) in enumerate(train_loader):
            batch_num += 1
            # print(flux.shape)
            optimizer.zero_grad()
            out = model(flux)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
            Loss.append(loss.item())
            batch.append(batch_num)

        model.eval()
        all_y_true = []
        all_y_pred = []
        for i, (flux, labels) in enumerate(test_loader):
            out = model(flux)
            out = out.argmax(dim=1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(out.detach().cpu().numpy())

        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
        # 计算整体的正确率
        print("Overall Accuracy: {:.2f}".format(overall_accuracy))
        # 计算召回率
        recall = recall_score(all_y_true, all_y_pred)
        print("Overall Recall: {:.2f}".format(recall))
        acc.append(overall_accuracy)
        rec.append(recall)

        conf_matrix = confusion_matrix(all_y_true, all_y_pred)
        plt.matshow(conf_matrix, cmap=plt.cm.Blues)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
        plt.xticks(ticks=[0, 1, 2], labels=class_names)
        plt.yticks(ticks=[0, 1, 2], labels=class_names)
        plt.title("Overall Accuracy: {:.2f}".format(overall_accuracy))
        plt.ylabel('Real label')
        plt.xlabel('Predicted label')
        plt.colorbar()
        plt.show(block=True)
        plt.savefig('./conf_matrix/' + str(epoch) + '.png')


        torch.save(model, "./model/lamost_model_{}.pkl".format(str(epoch)))

    plt.plot(batch, Loss, color='green', linestyle='solid', label='Train_Loss')
    plt.legend(loc='upper right')
    plt.title('Loss during lamostModel training')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig('./loss/' + str(epochs) + '.png')
    plt.close()

    plt.plot(range(1, epochs+1),acc, color='black', linestyle='solid', label='ACC')
    plt.plot(range(1, epochs + 1), rec, color='green', linestyle='solid', label='REC')
    plt.legend(loc='upper right')
    plt.title('Accuracy and Recall')
    plt.xlabel('epoch')
    plt.ylabel('percent')
    plt.savefig('./acc_rec/' + str(epochs) + '.png')



if __name__ == "__main__":
    c = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
    sample_size = 3500

    filepath = listdir('./data/train_data_norm')
    GALAXY_flux = np.load('./data/train_data_norm/GALAXY_flux.npy')
    QSO_flux = np.load('./data/train_data_norm/QSO_flux.npy')
    STAR_flux = np.load('./data/train_data_norm/STAR_flux.npy')

    # print(GALAXY_flux.shape, QSO_flux.shape, STAR_flux.shape) #(3427, 3000) (1011, 3000) (95562, 3000)
    GALAXY_flux_ = sample_expansion(GALAXY_flux, size=sample_size)
    QSO_flux_ = sample_expansion(QSO_flux, size=sample_size)

    epochs = 6

    eval(GALAXY_flux_, QSO_flux_, STAR_flux, epochs=epochs)






