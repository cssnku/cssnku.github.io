# README for VersaPSE dataset format

## Training data format

For training VersaPSE on sister passwords of users, the format is:

src \t trg\n

where \t denotes a tab (ignore the white spaces around it), src and trg are source password and target password, respectively. An example can be found in pseudo_train_data-sister.txt.

For training VersaPSE on users' passwords and similar popular passwords, the format is:

src \t trg \t count\n

where \t denotes a tab (ignore the white spaces around it), src is the matched popular password, and trg is the user's password. An example can be found in pseudo_train_data-popular.txt.

## Test data format

For targeted evaluation, the data format is:

src \t trg \t pop\n

where \t denotes a tab (ignore the white spaces around it), src is the source (sister) password, trg is the user's password to
be evaluated, and pop denotes a most similar popular password. An example can be found in pseudo_test_data-targeted.txt.

For untargeted evaluation, the data format is:

src \t trg \t count\n

where \t denotes a tab (ignore the white spaces around it), src is the matched popular password, trg is the user's password to
be evaluated, and count denotes frequency (used for WSpearman calculation). An example can be found in 
pseudo_test_data-untargeted.txt.


