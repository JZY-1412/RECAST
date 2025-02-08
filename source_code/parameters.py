class parameters:
    def __init__(self, dataset):
        if dataset == "porto":
            # model
            self.batch_size = 256
            self.vocab_size = 10521
            self.embed_size = 128
            self.hidden_size = 256
            self.num_layers = 2
            self.dropout = 0.3

            # training
            self.learning_rate = 0.001
            self.epoch_num = 100
            self.max_grad_norm = 5.0

            # loss weight
            self.recon_loss_weight = 10
            self.src_recon_loss_weight = 10
            self.route_recon_loss_weight = 10
            self.gaussian_loss_weight = 100
            self.cate_loss_weight = 100
            self.rdn_loss_weight = 1

            # evaluation
            self.eval_batch_size = 256
            self.positive_label = 1
        
        elif dataset == "beijing":
            # model
            self.batch_size = 256
            self.vocab_size = 71836
            self.embed_size = 128
            self.hidden_size = 256
            self.num_layers = 2
            self.dropout = 0.3

            # training
            self.learning_rate = 0.001
            self.epoch_num = 100
            self.max_grad_norm = 5.0

            # loss weight
            self.recon_loss_weight = 10
            self.src_recon_loss_weight = 10
            self.route_recon_loss_weight = 10
            self.gaussian_loss_weight = 100
            self.cate_loss_weight = 100
            self.rdn_loss_weight = 1

            # evaluation
            self.eval_batch_size = 256
            self.positive_label = 1
