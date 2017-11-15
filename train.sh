#!/bin/sh

screen -dm bash -c 'python train_optimization.py; exec sh'
sleep 5
screen -dm bash -c 'python train_evaluator.py; exec sh'
screen -dm bash -c 'python train_self_play.py; exec sh'
