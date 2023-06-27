from tensorboard import program

print("开始Tensorboard")
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'runs'])
tb.main()