#%%
import numpy as np
import pickle
import easydict
import json
from nn.optim import (
    SGD,
    Adagrad,
    RMSprop,
    Adadelta,
    Adam
)
from nn.utils import (
    Load_dataset,
    Dataloader,
    grad_clip_norm,
    EarlyStopping,
)
from MultiLayerNet import MultiLayerNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
#%%
def main():
    # 1. 데이터 불러오기
    print("Data 불러오기")
    X, y = Load_dataset()

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
    X.astype(np.float32),
    y.astype(np.float32),
    test_size=(1 / 7.)
    )
    #%%
    # data shape 확인
    # print(x_train.shape)
    # print(y_train.shape)
    #%%
    # 2. argument 설정
    args = easydict.EasyDict({
        "batch_size" : 100,
        "dataset" : "mnist_784",
        "epochs" : 100,
        "input_size" : 784,
        "hidden_layer_num" : 1,
        "hidden_size" : [100],
        "lr" : 1e-1,
        "output_size" : 10,
    })
    #%%
    # 3. 모델 및 optimizer 선언
    model = MultiLayerNet(input_size=args.input_size,
                            hidden_layer_num=args.hidden_layer_num,
                            hidden_size_list=args.hidden_size,
                            output_size=args.output_size,
                            )
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0)
    # optimizer = Adagrad(model.parameters(), lr=args.lr)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    # optimizer = Adadelta(model.parameters(), lr=args.lr, alpha=0.9)
    # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    es = EarlyStopping(patience=3, improved_std=0.0001)
    flag = True

    dataloader, steps_per_epoch = Dataloader(x_train, y_train, batch_size=args.batch_size, train=True)

    step = 0
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0
    #%%
    # 4. 학습 기록 인자
    model_name = "MLP"
    optimizer_name = "SGD"
    file_name = f"dataset={args.dataset}_lr={args.lr}_model={model_name}_optimizer={optimizer_name}_epochs={args.epochs}_batch={args.batch_size}_hidden={args.hidden_size}.json"
    datas = []
    #%%
    # 5. 학습
    for epoch in range(args.epochs):
        print("Epoch", epoch)

        for x_batch, y_batch in tqdm(dataloader):
            loss = model(x_batch, y_batch)

            grads = model.gradient()
            grads = grad_clip_norm(grads, 1)
            optimizer.step(grads)

            train_losses.append(loss)

            step += 1

            if step % steps_per_epoch == 0:
                train_acc = model.accuracy(x_train, y_train)
                test_acc = model.accuracy(x_test, y_test)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                if es.step(test_acc):
                    print(f"Train Accuracy: {train_acc}")
                    print(f"Test Accuracy: {test_acc}")
                    print(f"Loss: {loss}")

                    if test_acc >= best_acc:
                        best_acc = test_acc
                        with open('best_model.pickle', 'wb') as f:
                            pickle.dump(model.params, f)
                        print("Best model saved...")
                else:
                    flag = False

        datas.append({
            "epoch" : epoch,
            "train loss" : loss,
            "train acc" : train_acc,
            "test acc" : test_acc,
        })

        if not flag:
            print(f"Train Accuracy: {train_acc}")
            print(f"Test Accuracy: {test_acc}")
            print(f"loss: {loss}")
            print("Early stopping으로 학습을 종료합니다.")
            break

        with open(file_name, "w") as f:
            json.dump(datas, f, indent='\t')

#%%
    # 저장된 parameter를 가져올 새로운 모델 생성
    my_net = MultiLayerNet(args.input_size, args.hidden_layer_num, args.hidden_size, output_size=args.output_size)
    my_net.load_model("best_model.pickle")
#%%
    # parameter load 확인
    print("-Loading 모델 정확도 출력-")
    print(my_net.accuracy(x_test, y_test))
#%%
    # model inference
    infer, answer = my_net.inference(x_test, y_test, 10)
    print("-모델 inference 출력-")
    print(f"모델 inference: {infer}")
    print(f"정답: {answer}")
#%%
if __name__ == "__main__":
    main()