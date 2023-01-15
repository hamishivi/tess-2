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

