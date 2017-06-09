#GE-Mut-CNV-Methylation
macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_test_pIC50_fold_1_.sdm --output-prefix rst/fold1_genomic_k100_test --row-features ../features/binary/GE.sdm --row-features ../features/binary/Mut.sbm --row-features ../features/binary/Methy.sbm --row-features ../features/binary/CNV.sbm

macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_cal_pIC50_fold_1_.sdm --output-prefix rst/fold1_genomic_k100_cal --row-features ../features/binary/GE.sdm --row-features ../features/binary/Mut.sbm --row-features ../features/binary/Methy.sbm --row-features ../features/binary/CNV.sbm

#ECFP-Target
macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_test_pIC50_fold_1_.sdm --output-prefix rst/fold1_chemical_k100_test --col-features ../features/binary/ECFP.sbm --col-features ../features/binary/Target.sbm

macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_cal_pIC50_fold_1_.sdm --output-prefix rst/fold1_chemical_k100_cal  --col-features ../features/binary/ECFP.sbm --col-features ../features/binary/Target.sbm


#All
macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_test_pIC50_fold_1_.sdm --output-prefix rst/fold1_all_k100_test --row-features ../features/binary/GE.sdm --row-features ../features/binary/Mut.sbm --row-features ../features/binary/Methy.sbm --row-features ../features/binary/CNV.sbm --col-features ../features/binary/ECFP.sbm --col-features ../features/binary/Target.sbm

macau_nompi --train both_dim_train_pIC50_fold_1_.sdm --row-prior spikeandslab --col-prior spikeandslab --burnin 1600 --nsamples 401 \
--num-latent 100 --output-freq 400 --test both_dim_cal_pIC50_fold_1_.sdm --output-prefix rst/fold1_all_k100_cal  --row-features ../features/binary/GE.sdm --row-features ../features/binary/Mut.sbm --row-features ../features/binary/Methy.sbm --row-features ../features/binary/CNV.sbm --col-features ../features/binary/ECFP.sbm --col-features ../features/binary/Target.sbm
