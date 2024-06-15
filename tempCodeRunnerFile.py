        self.create_test_datasets()
        true_count = 0
        i = 0
        for true_y, tst_img in zip(self.test_labels, self.test_images):
            pred, _ = self.recognise_face(tst_img)
            # print(f"image {i}, true y {true_y}, pred {pred}\n\n")
            if true_y == pred:
                true_count += 1
            i += 1
        acc = true_count / self.test_labels.shape[0]