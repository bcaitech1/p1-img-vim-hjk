import os, sys
import numpy as np

import wandb
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_with_val(model_name, 
                   k, 
                   model, 
                   criterion, 
                   optimizer, 
                   train_loader, 
                   val_loader, 
                   scheduler, 
                   num_epochs, 
                   train_log_interval, 
                   val_set, 
                   mixed_precision_scaler, 
                   batch_size):
    
    print("\nTraining Start\n")
    patience = 10
    counter = 0
    best_val_acc = 0
    best_val_loss = np.inf
    
    # Tensorboard 로그를 저장할 경로 지정
    logger = SummaryWriter(log_dir=f"logdir/{model_name}")
    
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        loss_value = 0
        mask_loss_value = 0
        gender_loss_value = 0
        age_loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=True):
                img, label, mask, gender, age = train_batch
                img = img.to(device)
                label = label.to(device)

                logit, loss = model(img, label, criterion)
            
            pred = torch.argmax(logit, dim=-1)                        
            mixed_precision_scaler.scale(loss).backward()
            mixed_precision_scaler.step(optimizer)
            mixed_precision_scaler.update()
            optimizer.zero_grad()
            
            loss_value += loss.item()
            matches += (pred == label).sum().item()
            
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval
                train_acc = matches / batch_size / train_log_interval
                #current_lr = scheduler.get_lr()
                print(
                    f"Epoch[{epoch + 1}/{num_epochs}]({idx + 1}/{len(train_loader)})\n"
                    f"training loss {train_loss:.4f} || training accuracy {train_acc:4.2%}"
                )
                loss_value = 0
                matches = 0
                
                # Tensorboard 학습 단계에서 Loss, Accuracy 로그 저장
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                
                # wandb 학습 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "Train loss": train_loss,
                    "Train acc" : train_acc
                })
                
        # 각 에폭의 마지막 input 이미지로 grid view 생성
        img_grid = torchvision.utils.make_grid(img)
        # Tensorboard에 train input 이미지 기록
        logger.add_image(f'{epoch}_train_input_img', img_grid, epoch)
        
        with torch.no_grad():
            print("\nValidation Start\n")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            for val_batch in val_loader:
                img, label, mask, gender, age = val_batch
                img = img.to(device)
                label = label.to(device)

                logit, loss = model(img, label, criterion)
            
                pred = torch.argmax(logit, dim=-1)

                loss_item = loss.item()
                acc_item = (label == pred).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / val_set

            # Callback1 : validation accuracy가 향상될수록 모델을 저장합니다.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                os.makedirs(f'./checkpoint/{model_name}', exist_ok=True)
                torch.save(model.state_dict(), f"checkpoint/{model_name}/{k}_fold_{epoch:03}_accuracy_{val_acc:4.2%}.pth")
                best_val_acc = val_acc
            else:
                counter += 1

            # Callback2 : patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > patience:
                print("Early Stopping...")
                break

            print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}\n"
            )
            # Tensorboard 검증 단계에서 Loss, Accuracy 로그 저장
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)

            # wandb 검증 단계에서 Loss, Accuracy 로그 저장
            wandb.log({
                "Valid loss": val_loss,
                "Valid acc" : val_acc
            })
                
        scheduler.step(val_loss)

def multi_label_train_with_val(model_name, 
                               k, 
                               model, 
                               criterion, 
                               optimizer, 
                               train_loader, 
                               val_loader, 
                               scheduler, 
                               num_epochs, 
                               train_log_interval, 
                               val_set, 
                               mixed_precision_scaler, 
                               batch_size):
    
    print("\nTraining Start\n")
    patience = 10
    counter = 0
    best_val_acc = 0
    best_val_loss = np.inf
    
    # Tensorboard 로그를 저장할 경로 지정
    logger = SummaryWriter(log_dir=f"logdir/{model_name}")
    
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        
        loss_value = 0
        mask_loss_value = 0
        gender_loss_value = 0
        age_loss_value = 0
        
        matches = 0
        mask_matches = 0
        gender_matches = 0
        age_matches = 0
        
        for idx, train_batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=True):
                img, label, mask, gender, age = train_batch                
                img = img.to(device)
                label = label.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)

                mask_logit, gender_logit, age_logit = model(img)
                mask_loss = criterion(mask_logit, mask)
                gender_loss = criterion(gender_logit, gender)
                age_loss = criterion(age_logit, age)
                
                loss = mask_loss + gender_loss + age_loss
            
            pred_mask = torch.argmax(mask_logit, dim=-1)
            pred_gender = torch.argmax(gender_logit, dim=-1)
            pred_age = torch.argmax(age_logit, dim=-1)
                
            pred = pred_mask * 6 + pred_gender * 3 + pred_age
            
            mixed_precision_scaler.scale(loss).backward()
            mixed_precision_scaler.step(optimizer)
            mixed_precision_scaler.update()
            optimizer.zero_grad()
            
            loss_value += loss.item()
            matches += (pred == label).sum().item()
            
            mask_loss_value += mask_loss.item()
            gender_loss_value += gender_loss.item()
            age_loss_value += age_loss.item()
            
            mask_matches += (pred_mask == mask).sum().item()
            gender_matches += (pred_gender == gender).sum().item()
            age_matches += (pred_age == age).sum().item()
            
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval                
                train_acc = matches / batch_size / train_log_interval
                
                m_loss = mask_loss_value / train_log_interval
                g_loss = gender_loss_value / train_log_interval
                a_loss = age_loss_value / train_log_interval
                
                m_acc = mask_matches / batch_size / train_log_interval
                g_acc = gender_matches / batch_size / train_log_interval
                a_acc = age_matches / batch_size / train_log_interval
                
                #current_lr = scheduler.get_lr()
                print(
                    f"[{k + 1} Fold]\n"
                    f"Epoch[{epoch + 1}/{num_epochs}]({idx + 1}/{len(train_loader)})\n"
                    f"mask loss {m_loss:.4f} || gender loss {g_loss:.4f} || age loss {a_loss:.4f} || "
                    f"mask acc {m_acc:.4f} || gender acc {g_acc:.4f} || age acc {a_acc:.4f}\n"
                    f"training loss {train_loss:.4f} || training accuracy {train_acc:4.2%}"
                )
                loss_value = 0
                matches = 0
                
                mask_loss_value = 0
                gender_loss_value = 0
                age_loss_value = 0
                mask_matches = 0
                gender_matches = 0
                age_matches = 0
                
                # Tensorboard 학습 단계에서 Loss, Accuracy 로그 저장
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Mask/loss", m_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Mask/accuracy", m_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Gender/loss", g_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Gender/accuracy", g_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Age/loss", a_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Age/accuracy", a_acc, epoch * len(train_loader) + idx)
                
                # wandb 학습 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "Train loss": train_loss,
                    "Train acc" : train_acc,
                    "Mask loss": m_loss,
                    "Mask acc" : m_acc,
                    "Gender loss": g_loss,
                    "Gender acc" : g_acc,
                    "Age loss": a_loss,
                    "Age acc" : a_acc
                })
                
        # 각 에폭의 마지막 input 이미지로 grid view 생성
        img_grid = torchvision.utils.make_grid(img)
        # Tensorboard에 train input 이미지 기록
        logger.add_image(f'{epoch}_train_input_img', img_grid, epoch)
        
        with torch.no_grad():
            print("\nValidation Start\n")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            mask_loss_items = []
            mask_acc_items = []
            gender_loss_items = []
            gender_acc_items = []
            age_loss_items = []
            age_acc_items = []
            
            for val_batch in val_loader:
                img, label, mask, gender, age = val_batch
                img = img.to(device)
                label = label.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)

                mask_logit, gender_logit, age_logit = model(img)
                mask_loss = criterion(mask_logit, mask)
                gender_loss = criterion(gender_logit, gender)
                age_loss = criterion(age_logit, age)
                
                loss = mask_loss + gender_loss + age_loss
            
                pred_mask = torch.argmax(mask_logit, dim=-1)
                pred_gender = torch.argmax(gender_logit, dim=-1)
                pred_age = torch.argmax(age_logit, dim=-1)

                pred = pred_mask * 6 + pred_gender * 3 + pred_age

                loss_item = loss.item()
                acc_item = (label == pred).sum().item()
                
                mask_loss_item = mask_loss.item()
                gender_loss_item = gender_loss.item()
                age_loss_item = age_loss.item()

                mask_acc_item = (pred_mask == mask).sum().item()
                gender_acc_item = (pred_gender == gender).sum().item()
                age_acc_item = (pred_age == age).sum().item()
                
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                mask_loss_items.append(mask_loss_item)
                mask_acc_items.append(mask_acc_item)
                gender_loss_items.append(gender_loss_item)
                gender_acc_items.append(gender_acc_item)
                age_loss_items.append(age_loss_item)
                age_acc_items.append(age_acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / val_set
            
            val_mask_loss = np.sum(mask_loss_items) / len(val_loader)
            val_mask_acc = np.sum(mask_acc_items) / val_set
            val_gender_loss = np.sum(gender_loss_items) / len(val_loader)
            val_gender_acc = np.sum(gender_acc_items) / val_set
            val_age_loss = np.sum(age_loss_items) / len(val_loader)
            val_age_acc = np.sum(age_acc_items) / val_set

            # Callback1 : validation accuracy가 향상될수록 모델을 저장합니다.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model..")
                os.makedirs(f'./checkpoint/{model_name}', exist_ok=True)
                torch.save(model.state_dict(), f"checkpoint/{model_name}/{k}_fold_{epoch:03}_accuracy_{val_acc:4.2%}.pth")
                best_val_acc = val_acc
            else:
                counter += 1

            # Callback2 : patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > patience:
                print("Early Stopping...")
                break

            print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || \n"
                    f"mask acc : {val_mask_acc:4.2%}, mask loss: {val_mask_loss:4.2} || "
                    f"gender acc : {val_gender_acc:4.2%}, gender loss: {val_gender_loss:4.2} || "
                    f"age acc : {val_age_acc:4.2%}, age loss: {val_age_loss:4.2} || \n"
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}\n"
            )
            # Tensorboard 검증 단계에서 Loss, Accuracy 로그 저장
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/Mask loss", val_mask_loss, epoch)
            logger.add_scalar("Val/Mask accuracy", val_mask_acc, epoch)
            logger.add_scalar("Val/Gender loss", val_gender_loss, epoch)
            logger.add_scalar("Val/Gender accuracy", val_gender_acc, epoch)
            logger.add_scalar("Val/Age loss", val_age_loss, epoch)
            logger.add_scalar("Val/Age accuracy", val_age_acc, epoch)

            # wandb 검증 단계에서 Loss, Accuracy 로그 저장
            wandb.log({
                "Valid loss": val_loss,
                "Valid acc" : val_acc,
                "Mask loss": val_mask_loss,
                "Mask acc" : val_mask_acc,
                "Gender loss": val_gender_loss,
                "Gender acc" : val_gender_acc,
                "Age loss": val_age_loss,
                "Age acc" : val_age_acc
            })
        
        
def multi_label_train(model_name, 
                      model, 
                      criterion, 
                      optimizer, 
                      train_loader, 
                      scheduler, 
                      num_epochs, 
                      train_log_interval, 
                      mixed_precision_scaler, 
                      batch_size):
    
    print("\nTraining Start\n")
    patience = 10
    counter = 0
    best_val_acc = 0
    best_val_loss = np.inf
    
    # Tensorboard 로그를 저장할 경로 지정
    logger = SummaryWriter(log_dir=f"logdir/{model_name}")
    
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        
        loss_value = 0
        mask_loss_value = 0
        gender_loss_value = 0
        age_loss_value = 0
        
        matches = 0
        mask_matches = 0
        gender_matches = 0
        age_matches = 0
        
        for idx, train_batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=True):
                img, label, mask, gender, age = train_batch                
                img = img.to(device)
                label = label.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)

                mask_logit, gender_logit, age_logit = model(img)
                mask_loss = criterion(mask_logit, mask)
                gender_loss = criterion(gender_logit, gender)
                age_loss = criterion(age_logit, age)
                
                loss = mask_loss + gender_loss + age_loss
            
            pred_mask = torch.argmax(mask_logit, dim=-1)
            pred_gender = torch.argmax(gender_logit, dim=-1)
            pred_age = torch.argmax(age_logit, dim=-1)
                
            pred = pred_mask * 6 + pred_gender * 3 + pred_age
            
            optimizer.zero_grad()
            mixed_precision_scaler.scale(loss).backward()
            mixed_precision_scaler.step(optimizer)
            mixed_precision_scaler.update()        
            
            loss_value += loss.item()
            matches += (pred == label).sum().item()
            
            mask_loss_value += mask_loss.item()
            gender_loss_value += gender_loss.item()
            age_loss_value += age_loss.item()
            
            mask_matches += (pred_mask == mask).sum().item()
            gender_matches += (pred_gender == gender).sum().item()
            age_matches += (pred_age == age).sum().item()
            
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval                
                train_acc = matches / batch_size / train_log_interval
                current_lr = scheduler.get_last_lr()
                
                m_loss = mask_loss_value / train_log_interval
                g_loss = gender_loss_value / train_log_interval
                a_loss = age_loss_value / train_log_interval
                
                m_acc = mask_matches / batch_size / train_log_interval
                g_acc = gender_matches / batch_size / train_log_interval
                a_acc = age_matches / batch_size / train_log_interval
                
                print(
                    f"Epoch[{epoch + 1}/{num_epochs}]({idx + 1}/{len(train_loader)})\n"
                    f"mask loss {m_loss:.4f} || gender loss {g_loss:.4f} || age loss {a_loss:.4f} || "
                    f"mask acc {m_acc:.4f} || gender acc {g_acc:.4f} || age acc {a_acc:.4f}\n"
                    f"training loss {train_loss:.4f} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                loss_value = 0
                matches = 0
                
                mask_loss_value = 0
                gender_loss_value = 0
                age_loss_value = 0
                mask_matches = 0
                gender_matches = 0
                age_matches = 0
                
                # Tensorboard 학습 단계에서 Loss, Accuracy 로그 저장
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Mask/loss", m_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Mask/accuracy", m_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Gender/loss", g_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Gender/accuracy", g_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Age/loss", a_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Age/accuracy", a_acc, epoch * len(train_loader) + idx)
                
                # wandb 학습 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "Train loss": train_loss,
                    "Train acc" : train_acc,
                    "Mask loss": m_loss,
                    "Mask acc" : m_acc,
                    "Gender loss": g_loss,
                    "Gender acc" : g_acc,
                    "Age loss": a_loss,
                    "Age acc" : a_acc
                })
        scheduler.step()                
       
        os.makedirs(f'./checkpoint/{model_name}', exist_ok=True)
        torch.save(model.state_dict(), f"checkpoint/{model_name}/{model_name}_{epoch:03}epoch.pth")