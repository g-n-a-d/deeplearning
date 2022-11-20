import torch

class First_Order_Model(torch.nn.Module):
    def __init__(self, kp_extractor, dense_motion, generator, discriminator, train_params):
        super().__init__()
        self.kp_extractor = kp_extractor
        self.dense_motion = dense_motion
        self.generator = generator
        self.criterion = 
        self.optimizer = 
        # self.discriminator = discriminator
        # self.train_params = train_params
        # self.scales = train_params['scales']
        # self.disc_scales = self.discriminator.scales
        # self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        # if torch.cuda.is_available():
        #     self.pyramid = self.pyramid.cuda()

        # self.loss_weights = train_params['loss_weights']

        # if sum(self.loss_weights['perceptual']) != 0:
        #     self.vgg = Vgg19()
        #     if torch.cuda.is_available():
        #         self.vgg = self.vgg.cuda()

    def forward(self, frame_source, frame_driving):
        kp_source = self.kp_extractor(frame_source) #kp(b, num_kp, 2), jacobian()
        kp_driving = self.kp_extractor(frame_driving) #kp(b, num_kp, 2), jacobian()
        motion = self.dense_motion(frame_source, kp_source, kp_driving) #motion(b, h, w, 2), occlusion(b, 1, h, w)
        frame_generated = self.generator(frame_source, motion) #(b, num_channels, h, w)
        outs = {
            'kp_source':kp_source,
            'kp_driving':kp_driving,
            'motion':motion,
            'frame_generated':frame_generated
        }
        # generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        # loss_values = {}

        # pyramide_real = self.pyramid(x['driving'])
        # pyramide_generated = self.pyramid(generated['prediction'])

        # if sum(self.loss_weights['perceptual']) != 0:
        #     value_total = 0
        #     for scale in self.scales:
        #         x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
        #         y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

        #         for i, weight in enumerate(self.loss_weights['perceptual']):
        #             value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
        #             value_total += self.loss_weights['perceptual'][i] * value
        #         loss_values['perceptual'] = value_total

        # if self.loss_weights['generator_gan'] != 0:
        #     discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        #     discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        #     value_total = 0
        #     for scale in self.disc_scales:
        #         key = 'prediction_map_%s' % scale
        #         value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
        #         value_total += self.loss_weights['generator_gan'] * value
        #     loss_values['gen_gan'] = value_total

        #     if sum(self.loss_weights['feature_matching']) != 0:
        #         value_total = 0
        #         for scale in self.disc_scales:
        #             key = 'feature_maps_%s' % scale
        #             for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
        #                 if self.loss_weights['feature_matching'][i] == 0:
        #                     continue
        #                 value = torch.abs(a - b).mean()
        #                 value_total += self.loss_weights['feature_matching'][i] * value
        #             loss_values['feature_matching'] = value_total

        # if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
        #     transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
        #     transformed_frame = transform.transform_frame(x['driving'])
        #     transformed_kp = self.kp_extractor(transformed_frame)

        #     generated['transformed_frame'] = transformed_frame
        #     generated['transformed_kp'] = transformed_kp

        #     ## Value loss part
        #     if self.loss_weights['equivariance_value'] != 0:
        #         value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
        #         loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

        #     ## jacobian loss part
        #     if self.loss_weights['equivariance_jacobian'] != 0:
        #         jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
        #                                             transformed_kp['jacobian'])

        #         normed_driving = torch.inverse(kp_driving['jacobian'])
        #         normed_transformed = jacobian_transformed
        #         value = torch.matmul(normed_driving, normed_transformed)

        #         eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

        #         value = torch.abs(eye - value).mean()
        #         loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return outs

    def learn(self, data_loader, num_epochs=10):
        for epoch in range(num_epochs):
            for x in data_loader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)