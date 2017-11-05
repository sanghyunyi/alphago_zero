#!/bin/sh

screen -d -m python train_optimization.py
sleep 5
screen -d -m python train_evaluator.py
screen -d -m python train_self_play.py
