
# page features
PAGE_POPULARITY = 'v[0]' # Decimal Encoding. Defines the popularity or support for the source of the document.
PAGE_CHECKINGS = 'v[1]' # Decimal Encoding. Describes how many individuals so far visited this place. This feature is only associated with the places eg:some institution, place, theater etc.
PAGE_TALKING_ABOUT = 'v[2]' # Decimal Encoding. Defines the daily interest of individuals towards source of the document/ Post. The people who actually come back to the page, after liking the page. This include activities such as comments, likes to a post, shares, etc by visitors to the page.
PAGE_CATEGORY = 'v[3]' # Value Encoding. Defines the category of the source of the document eg: place, institution, brand etc.

# feature parameters. These features are aggregated by page, by calculating min, max, average, median and standard deviation of essential features.
MIN_COMMENTS_BEFORE_BASELINE = 'v[4]'
MAX_COMMENTS_BEFORE_BASELINE = 'v[5]'
AVERAGE_COMMENTS_BEFORE_BASELINE = 'v[6]'
MEDIAN_COMMENTS_BEFORE_BASELINE = 'v[7]'
STANDARD_DEVIATION_COMMENTS_BEFORE_BASELINE = 'v[8]'

MIN_COMMENTS_IN_LAST_24 = 'v[9]'
MAX_COMMENTS_IN_LAST_24 = 'v[10]'
AVERAGE_COMMENTS_IN_LAST_24 = 'v[11]'
MEDIAN_COMMENTS_IN_LAST_24 = 'v[12]'
STANDARD_DEVIATION_COMMENTS_IN_LAST_24 = 'v[13]'

MIN_COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[14]'
MAX_COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[15]'
AVERAGE_COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[16]'
MEDIAN_COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[17]'
STANDARD_DEVIATION_COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[18]'

MIN_COMMENTS_IN_FIRST_24 = 'v[19]'
MAX_COMMENTS_IN_FIRST_24 = 'v[20]'
AVERAGE_COMMENTS_IN_FIRST_24 = 'v[21]'
MEDIAN_COMMENTS_IN_FIRST_24 = 'v[22]'
STANDARD_DEVIATION_COMMENTS_IN_FIRST_24 = 'v[23]'


MIN_LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[24]'
MAX_LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[25]'
AVERAGE_LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[26]'
MEDIAN_LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[27]'
STANDARD_DEVIATION_LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[28]'

# essential features
COMMENTS_BEFORE_BASELINE = 'v[29]' # Decimal Encoding. The total number of comments before selected base date/time.
COMMENTS_IN_LAST_24 = 'v[30]' # Decimal Encoding. The number of comments in last 24 hours, relative to base date/time.
COMMENTS_IN_LAST_48_BEFORE_LAST_24 = 'v[31]' # Decimal Encoding. The number of comments in last 48 to last 24 hours relative to base date/time.
COMMENTS_IN_FIRST_24 = 'v[32]' # Decimal Encoding. The number of comments in the first 24 hours after the publication of post but before base date/time.
LAST_24_MINUS_LAST_48_BEFORE_24 = 'v[33]' # Decimal Encoding. The difference between 30 and 31

# other features
BASE_TIME = 'v[34]' # Decimal(0-71) Encoding. Selected time in order to simulate the scenario.
POST_LENGTH = 'v[35]' # Decimal Encoding. Character count in the post.
POST_SHARE_COUNT = 'v[36]' # Decimal Encoding. This features counts the no of shares of the post, that how many peoples had shared this post on to their timeline.
POST_PROMOTION_STATUS = 'v[37]' # Binary Encoding. To reach more people with posts in News Feed, individual promote their post and this features tells that whether the post is promoted(1) or not(0).
H_LOCAL = 'v[38]' # Decimal(0-23) Encoding. This describes the H hrs, for which we have the target variable/ comments received.
POST_PUBLISHED_SUNDAY = 'v[39]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_MONDAY = 'v[40]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_TUESDAY = 'v[41]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_WEDNESDAY = 'v[42]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_THURSDAY = 'v[43]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_FRIDAY = 'v[44]' # Binary Encoding. This represents the day on which the post was published.
POST_PUBLISHED_SATURDAY = 'v[45]' # Binary Encoding. This represents the day on which the post was published.

BASE_DATETIME_SUNDAY = 'v[46]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_MONDAY = 'v[47]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_TUESDAY = 'v[48]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_WEDNESDAY = 'v[49]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_THURSDAY = 'v[50]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_FRIDAY = 'v[51]' # Binary Encoding. This represents the day on selected base Date/Time.
BASE_DATETIME_SATURDAY = 'v[52]' # Binary Encoding. This represents the day on selected base Date/Time.

TARGET = 'v[53]' # Decimal. The no of comments in next H hrs(H is given in Feature no 39).


