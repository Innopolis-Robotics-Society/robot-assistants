std_msgs/Header header
iros_custom_msgs/Statistics stats

bool algo_ok          # алгоритм не упал (try/except)
bool ok               # плата ок/не ок (логика сравнения с эталоном)

bool match            # true только во время формирования эталона
bool baseline_set
bool calibrating
string reason

string anchor_class
bool has_anchor
iros_custom_msgs/Detection2D anchor

iros_custom_msgs/Detection2D[] pred
