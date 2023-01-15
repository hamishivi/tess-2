

# download and prepare training data
wget https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2
tar -xvjf data-simplification.tar.bz2
mkdir wikilarge
cp data-simplification/wikilarge/wiki.full.aner.ori.train.src wikilarge/s1.train
cp data-simplification/wikilarge/wiki.full.aner.ori.train.dst wikilarge/s2.train
cp data-simplification/wikilarge/wiki.full.aner.ori.valid.src wikilarge/s1.dev
cp data-simplification/wikilarge/wiki.full.aner.ori.valid.dst wikilarge/s2.dev
cp data-simplification/wikilarge/wiki.full.aner.ori.test.src wikilarge/s1.test
cp data-simplification/wikilarge/wiki.full.aner.ori.test.dst wikilarge/s2.test

# concat source and target of training data for pretraining
cp wikilarge/s1.train wikilarge/all_train
cat wikilarge/s2.train >> wikilarge/all_train

rm -rf data-simplification
rm data-simplification.tar.bz2

# download validation and test data
git clone https://github.com/cocoxu/simplification.git simp_temp

mkdir simplification simplification/test simplification/valid
#test
cp simp_temp/data/turkcorpus/test.8turkers.tok.norm simplification/test/norm
cp simp_temp/data/turkcorpus/test.8turkers.tok.simp simplification/test/simp
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.0 simplification/test/turk0
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.1 simplification/test/turk1
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.2 simplification/test/turk2
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.3 simplification/test/turk3
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.4 simplification/test/turk4
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.5 simplification/test/turk5
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.6 simplification/test/turk6
cp simp_temp/data/turkcorpus/test.8turkers.tok.turk.7 simplification/test/turk7

#valid
cp simp_temp/data/turkcorpus/tune.8turkers.tok.norm simplification/valid/norm
cp simp_temp/data/turkcorpus/tune.8turkers.tok.simp simplification/valid/simp
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.0 simplification/valid/turk0
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.1 simplification/valid/turk1
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.2 simplification/valid/turk2
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.3 simplification/valid/turk3
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.4 simplification/valid/turk4
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.5 simplification/valid/turk5
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.6 simplification/valid/turk6
cp simp_temp/data/turkcorpus/tune.8turkers.tok.turk.7 simplification/valid/turk7

rm -rf simp_temp

echo "Extraction of WikiLarge finished!"
