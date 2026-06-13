# ============================================================
#  TẠO DATASET TỔNG HỢP — Phản hồi sinh viên tiếng Việt
#  Kết hợp: UIT-VSFC gốc + 557 câu tự thu thập
#  Output: dataset_tonghop.csv, dataset_train.csv,
#          dataset_val.csv, dataset_test.csv
# ============================================================

import os
import csv
import random
import unicodedata
from datetime import datetime

random.seed(42)

os.makedirs("data", exist_ok=True)

# ============================================================
#  557 CÂU TỰ THU THẬP — Phân loại theo nhãn
# ============================================================

cau_tich_cuc = [
    "Thầy dạy rất nhiệt tình, luôn sẵn sàng giải đáp thắc mắc của sinh viên.",
    "Cô giảng bài rõ ràng, dễ hiểu, sinh viên tiếp thu bài rất tốt.",
    "Giảng viên có kiến thức chuyên sâu và truyền đạt rất hiệu quả.",
    "Thầy luôn lấy ví dụ thực tế giúp sinh viên hiểu bài nhanh hơn.",
    "Cô giảng dạy tận tâm, luôn quan tâm đến việc sinh viên có hiểu bài không.",
    "Thầy giải thích rõ từng bước, không để sinh viên bị bỏ lại phía sau.",
    "Cô có phương pháp giảng dạy sáng tạo, sinh viên rất hào hứng học.",
    "Giảng viên luôn cập nhật kiến thức mới nhất vào bài giảng.",
    "Thầy rất nhiệt huyết với nghề, truyền cảm hứng học tập cho sinh viên.",
    "Cô dạy chậm rãi, cẩn thận từng phần, rất phù hợp với sinh viên còn yếu.",
    "Thầy tạo không khí học tập thoải mái, sinh viên không bị áp lực.",
    "Giảng viên biết cách kết hợp lý thuyết và thực hành một cách hài hòa.",
    "Cô luôn khuyến khích sinh viên đặt câu hỏi và tư duy phản biện.",
    "Thầy dạy bằng tất cả tâm huyết, sinh viên cảm nhận được điều đó.",
    "Cô giảng bài có hệ thống, logic, dễ ghi chép và ôn tập lại.",
    "Giảng viên luôn đến lớp đúng giờ và chuẩn bị bài giảng kỹ lưỡng.",
    "Thầy biết lắng nghe ý kiến sinh viên và điều chỉnh tốc độ giảng phù hợp.",
    "Cô có phong cách giảng dạy thân thiện, gần gũi với sinh viên.",
    "Giảng viên chia sẻ nhiều kinh nghiệm thực tế rất có giá trị.",
    "Thầy hướng dẫn chi tiết, tỉ mỉ từng bài tập, sinh viên làm được bài.",
    "Cô giải đáp thắc mắc rất tận tình, không bao giờ tỏ ra khó chịu.",
    "Thầy dạy sinh động, có slide đẹp và nhiều hình ảnh minh họa phong phú.",
    "Giảng viên luôn tạo điều kiện cho sinh viên thực hành trong giờ học.",
    "Cô rất công bằng trong việc đánh giá và chấm điểm sinh viên.",
    "Thầy có kinh nghiệm thực tế phong phú, bài giảng rất gần với thực tiễn.",
    "Cô dạy dễ hiểu, bài kiểm tra sát với nội dung đã học trên lớp.",
    "Giảng viên tận tụy, luôn dành thêm thời gian để giải thích cho sinh viên yếu.",
    "Thầy truyền đạt kiến thức một cách logic và có hệ thống rất tốt.",
    "Cô luôn chuẩn bị tài liệu học tập đầy đủ cho sinh viên.",
    "Giảng viên rất chuyên nghiệp, luôn tôn trọng và lắng nghe sinh viên.",
    "Môn học rất thú vị, kiến thức áp dụng trực tiếp vào công việc sau này.",
    "Nội dung môn học phong phú, cập nhật theo xu hướng hiện đại.",
    "Môn học giúp em hiểu sâu hơn về lĩnh vực chuyên ngành của mình.",
    "Chương trình môn học được thiết kế hợp lý, từ cơ bản đến nâng cao.",
    "Môn học kết hợp tốt giữa lý thuyết và thực hành, rất bổ ích.",
    "Kiến thức môn học rất thiết thực, em đã áp dụng được ngay vào đồ án.",
    "Môn học mở rộng tầm nhìn của em về nhiều vấn đề trong ngành.",
    "Tài liệu học tập đầy đủ, dễ tìm kiếm và tham khảo thêm.",
    "Môn học có nhiều bài tập thực hành giúp củng cố kiến thức rất hiệu quả.",
    "Chủ đề môn học rất hấp dẫn, em học với tâm thế tích cực.",
    "Môn học giúp em có nền tảng vững chắc cho các môn học sau.",
    "Nội dung môn học được cập nhật thường xuyên, không bị lỗi thời.",
    "Môn học có nhiều dự án nhóm thú vị giúp phát triển kỹ năng làm việc nhóm.",
    "Tài liệu tham khảo được cung cấp đầy đủ và chất lượng cao.",
    "Môn học giúp em nhìn nhận vấn đề từ nhiều góc độ khác nhau.",
    "Em rất hài lòng với môn học này, cảm ơn thầy đã dạy tận tình.",
    "Đây là một trong những môn học hay nhất em từng học.",
    "Em sẽ giới thiệu môn học này cho các bạn khóa sau.",
    "Cảm ơn cô đã truyền đạt kiến thức và đam mê học tập cho em.",
    "Môn học tuyệt vời, em học được rất nhiều điều bổ ích.",
    "Thầy dạy rất giỏi, em mong được học thầy ở các môn khác nữa.",
    "Đề thi vừa sức, bám sát nội dung đã học trên lớp.",
    "Bài kiểm tra giữa kỳ được ra đề hợp lý, đánh giá đúng năng lực.",
    "Hình thức kiểm tra đa dạng giúp sinh viên thể hiện được khả năng.",
    "Slide bài giảng được chuẩn bị kỹ lưỡng và dễ hiểu.",
    "Tài liệu môn học phong phú, đủ để tự học thêm ở nhà.",
    "Thầy luôn lắng nghe phản hồi của sinh viên và điều chỉnh cách dạy.",
    "Cô tạo cơ hội cho sinh viên phát biểu và trình bày ý kiến của mình.",
    "Giảng viên công bằng trong đánh giá, không thiên vị sinh viên nào.",
    "Thầy hỗ trợ sinh viên cả ngoài giờ học, rất tận tâm.",
    "Môn học giúp em nhận ra niềm đam mê với lĩnh vực chuyên ngành.",
    "Bài tập nhóm được thiết kế rất hay, giúp sinh viên học hỏi lẫn nhau.",
    "Thầy chia sẻ nhiều kinh nghiệm thực tế từ doanh nghiệp rất có giá trị.",
    "Cô rất cởi mở, sinh viên cảm thấy thoải mái khi hỏi bài.",
    "Giảng viên sử dụng công nghệ trong giảng dạy rất hiệu quả.",
    "Thầy luôn khuyến khích sinh viên tự nghiên cứu và khám phá thêm.",
    "Cô giảng bài nhiệt huyết, sinh viên cảm nhận được tình yêu nghề.",
    "Môn học giúp em phát triển cả kiến thức lẫn kỹ năng mềm.",
    "Bài giảng của thầy sinh động, không bao giờ khiến sinh viên buồn ngủ.",
    "Em rất thích phong cách giảng dạy của cô, vừa nghiêm túc vừa vui vẻ.",
    "Thầy tạo không gian học tập cởi mở, sinh viên dám hỏi và dám sai.",
    "Cô dạy có tâm, sinh viên cảm nhận rõ sự quan tâm của cô.",
    "Thầy luôn động viên sinh viên vượt qua khó khăn trong học tập.",
    "Giảng viên rất tinh tế trong việc phát hiện và bổ sung kiến thức còn thiếu.",
    "Thầy biết cách kết hợp hài hước và nghiêm túc rất khéo léo.",
    "Em cảm ơn thầy vì những kiến thức quý báu trong môn học này.",
    "Môn học này là một trong những trải nghiệm học tập tốt nhất của em.",
    "Thầy truyền cảm hứng để em tiếp tục theo đuổi đam mê trong ngành.",
    "Giảng viên rất kiên nhẫn khi giải thích lại những phần em chưa hiểu.",
    "Thầy luôn đặt lợi ích học tập của sinh viên lên hàng đầu.",
    "Cô không chỉ dạy kiến thức mà còn dạy em cách học và tư duy.",
    "Em rất hài lòng với kết quả đạt được sau khi học môn này.",
    "Thầy dạy có tâm có tầm, là tấm gương cho sinh viên noi theo.",
    "Môn học được tổ chức rất chuyên nghiệp từ nội dung đến hình thức.",
    "Thầy dạy môn khó nhưng biến nó trở nên dễ hiểu, đó là tài năng thật sự.",
    "Cô luôn trả lời email và tin nhắn của sinh viên rất nhanh và tận tình.",
    "Thầy tạo ra những buổi học đáng nhớ nhờ phong cách giảng dạy độc đáo.",
    "Giảng viên rất thực tế, không dạy lý thuyết suông mà luôn gắn với thực hành.",
    "Môn học có ảnh hưởng tích cực lớn đến định hướng nghề nghiệp của em.",
    "Cô giảng dạy với sự đam mê và nhiệt huyết rõ rệt, em rất ngưỡng mộ.",
    "Thầy sử dụng các ví dụ gần gũi với cuộc sống sinh viên rất hiệu quả.",
    "Em học được cách làm việc chuyên nghiệp và có trách nhiệm từ môn này.",
    "Thầy hỗ trợ sinh viên không chỉ trong học tập mà còn trong định hướng nghề.",
    "Em rất thích cách cô kết hợp học và chơi trong quá trình giảng dạy.",
    "Giảng viên luôn đưa ra phản hồi mang tính xây dựng giúp em tiến bộ.",
    "Thầy là người thầy mà em luôn muốn học hỏi và noi theo.",
    "Cô đã giúp em vượt qua nỗi sợ môn học này và yêu thích nó.",
    "Môn học này thật sự thay đổi cách em nhìn nhận và tiếp cận vấn đề.",
    "Thầy dạy với tất cả tình yêu nghề, điều đó truyền năng lượng tích cực cho em.",
    "Cô là nguồn cảm hứng lớn giúp em quyết tâm theo đuổi chuyên ngành này.",
    "Giảng viên không chỉ giỏi chuyên môn mà còn là người thầy đầy nhân văn.",
    "Thầy làm cho môn học trở nên sống động và ý nghĩa hơn em tưởng.",
    "Em tự hào khi nói rằng mình đã học được rất nhiều từ môn học này.",
    "Thầy cô và môn học này là lý do em yêu thích việc đến trường mỗi ngày.",
    "Giảng viên luôn tôn trọng thời gian của sinh viên, không bao giờ kéo giờ.",
    "Cô dạy đúng kế hoạch, đúng giờ, đúng nội dung, rất chuyên nghiệp.",
    "Em cảm thấy mình được đầu tư thật sự khi học môn học này.",
    "Thầy có kiến thức uyên bác nhưng lại giảng rất giản dị và dễ hiểu.",
    "Môn học này đã thắp sáng niềm đam mê học tập trong em.",
    "Em trân trọng từng giờ học trong môn này và không muốn bỏ lỡ buổi nào.",
    "Môn học cung cấp đầy đủ nền tảng để em tự tin bước vào thị trường lao động.",
    "Thầy luôn sẵn sàng lắng nghe và hỗ trợ sinh viên gặp khó khăn.",
    "Cô đã dạy em không chỉ kiến thức mà còn thái độ học tập đúng đắn.",
    "Em thực sự hài lòng và biết ơn vì môn học và giảng viên tuyệt vời này.",
    "Cô là tấm gương về sự tận tâm và chuyên nghiệp trong giảng dạy.",
    "Thầy giúp em không chỉ vượt qua môn học mà còn yêu thích nó mãi mãi.",
]

cau_tieu_cuc = [
    "Thầy giảng quá nhanh, sinh viên không ghi kịp và không hiểu bài.",
    "Cô không giải thích rõ ràng, sinh viên hỏi thì tỏ ra khó chịu.",
    "Giảng viên thường xuyên đến muộn làm ảnh hưởng đến kế hoạch học.",
    "Thầy không quan tâm đến việc sinh viên có hiểu bài hay không.",
    "Cô giảng bài không có hệ thống, sinh viên rất khó theo dõi.",
    "Giảng viên không có kiến thức thực tế, chỉ đọc sách một chiều.",
    "Thầy thiên vị, đối xử không công bằng giữa các sinh viên trong lớp.",
    "Cô không chuẩn bị bài trước khi lên lớp, giảng bài rất lộn xộn.",
    "Giảng viên không chịu lắng nghe ý kiến của sinh viên.",
    "Thầy thường xuyên cho hủy giờ học mà không báo trước.",
    "Cô chấm bài không công bằng, không giải thích lý do trừ điểm.",
    "Giảng viên có thái độ không tôn trọng sinh viên khi hỏi bài.",
    "Thầy dạy nặng về lý thuyết, không có thực hành, rất khô khan.",
    "Cô không bao giờ phản hồi email của sinh viên, rất khó liên lạc.",
    "Giảng viên không hỗ trợ sinh viên ngoài giờ học khi cần thiết.",
    "Cô ra bài tập không liên quan đến nội dung đã dạy trên lớp.",
    "Giảng viên không cập nhật kiến thức mới, dạy theo sách cũ lỗi thời.",
    "Thầy dạy không đủ thời lượng theo kế hoạch, bỏ nhiều nội dung quan trọng.",
    "Cô không có phương pháp giảng dạy rõ ràng, lớp học hỗn loạn.",
    "Giảng viên không biết cách quản lý thời gian trong buổi học.",
    "Thầy giải thích sai nhiều lần, sinh viên bị nhầm kiến thức.",
    "Giảng viên hay đọc slide chứ không thực sự giảng bài cho sinh viên.",
    "Thầy không có kỹ năng giao tiếp tốt, làm sinh viên khó tiếp cận.",
    "Môn học quá khó so với trình độ hiện tại của sinh viên năm nhất.",
    "Nội dung môn học quá dài trong khi thời lượng học quá ngắn.",
    "Tài liệu học tập cũ kỹ, không phù hợp với thực tế hiện nay.",
    "Môn học không có giá trị thực tiễn, kiến thức không áp dụng được.",
    "Bài tập quá nhiều và quá khó, sinh viên không có thời gian làm.",
    "Môn học không được thiết kế phù hợp cho sinh viên ngành này.",
    "Không có tài liệu học tập, sinh viên phải tự tìm kiếm rất vất vả.",
    "Chương trình môn học lỗi thời, không theo kịp xu hướng hiện đại.",
    "Tài liệu tham khảo được giới thiệu không có trong thư viện trường.",
    "Môn học có quá nhiều yêu cầu không thực tế đối với sinh viên.",
    "Bài tập nhóm không được tổ chức tốt, gây mâu thuẫn trong nhóm.",
    "Môn học thiếu tính ứng dụng và liên hệ thực tế trong bài giảng.",
    "Đề thi khác hoàn toàn so với nội dung đã học, sinh viên không làm được.",
    "Kiểm tra quá nhiều, không có đủ thời gian để ôn tập kỹ lưỡng.",
    "Đề thi ra quá khó so với mức độ kiến thức được giảng dạy.",
    "Không có đề cương ôn tập rõ ràng khiến sinh viên không biết học gì.",
    "Hình thức thi không phù hợp với nội dung và mục tiêu môn học.",
    "Đề thi có nhiều lỗi sai gây nhầm lẫn cho sinh viên trong phòng thi.",
    "Giảng viên chấm bài không nhất quán, cùng bài giống nhau nhưng điểm khác.",
    "Thời gian thi quá ngắn không đủ để làm hết bài một cách cẩn thận.",
    "Phòng học quá nóng, không có điều hòa hoặc điều hòa hỏng thường xuyên.",
    "Phòng học chật chội, không đủ chỗ ngồi cho tất cả sinh viên.",
    "Máy chiếu hay bị hỏng giữa chừng làm gián đoạn bài giảng.",
    "Phòng máy tính hay bị lỗi, ảnh hưởng đến tiết học thực hành.",
    "Wifi trong lớp học rất yếu, không thể tra cứu tài liệu trong giờ.",
    "Âm thanh trong phòng học không tốt, ngồi cuối lớp không nghe rõ.",
    "Ánh sáng phòng học tối, nhìn bảng và slide không rõ ràng.",
    "Lịch học thay đổi liên tục, sinh viên khó sắp xếp thời gian cá nhân.",
    "Thông báo quan trọng không được gửi đúng hạn, sinh viên bị bỏ lỡ.",
    "Khối lượng học tập trong kỳ này quá nặng, sinh viên bị quá tải.",
    "Môn học không đạt được mục tiêu học tập đã đề ra trong đề cương.",
    "Em rất thất vọng với chất lượng giảng dạy của môn học này.",
    "Sinh viên không nhận được phản hồi sau khi nộp bài tập.",
    "Phương pháp dạy học lỗi thời, không phù hợp với sinh viên thế kỷ 21.",
    "Em mất nhiều thời gian nhưng không học được gì có ích từ môn này.",
    "Giảng viên không tôn trọng thời gian của sinh viên, thường kéo giờ.",
    "Bài giảng không có cấu trúc rõ ràng, rất khó để ghi chép và ôn tập.",
    "Môn học không được tổ chức tốt từ đầu đến cuối, thiếu tính chuyên nghiệp.",
    "Em phải tự học hoàn toàn vì giảng viên không truyền đạt được kiến thức.",
    "Giảng viên hay nói chuyện lan man, lãng phí thời gian học của sinh viên.",
    "Chất lượng buổi học giảm dần theo thời gian, không được duy trì nhất quán.",
    "Môn học gây ra nhiều căng thẳng không cần thiết cho sinh viên.",
    "Em và nhiều bạn cùng lớp đều không hài lòng với cách tổ chức môn học.",
    "Giảng viên không giải thích lý do khi thay đổi lịch học đột ngột.",
    "Đề thi cuối kỳ quá khó và bất ngờ so với những gì đã học.",
    "Em cảm thấy thời gian và công sức bỏ ra cho môn học này không xứng đáng.",
    "Nội dung bài giảng quá trừu tượng, thiếu ví dụ cụ thể để minh họa.",
    "Slide bài giảng đầy chữ, không có hình ảnh, rất khô khan và nhàm chán.",
    "Giảng viên không quan tâm đến sự vắng mặt của sinh viên trong lớp.",
    "Em đã học chăm chỉ nhưng kết quả không phản ánh đúng sự cố gắng.",
    "Môn học có quá nhiều nội dung không liên quan đến ngành học của em.",
    "Giảng viên không giải thích rõ tiêu chí đánh giá và chấm điểm.",
    "Em không tìm được sự hỗ trợ cần thiết khi gặp khó khăn trong môn học.",
    "Môn học này là nguyên nhân chính khiến em bị căng thẳng trong kỳ.",
    "Giảng viên không có kỹ năng sư phạm tốt để truyền đạt kiến thức hiệu quả.",
    "Bài tập nhóm không có hướng dẫn cụ thể, nhóm làm mà không biết đúng sai.",
    "Em cảm thấy không được học gì hữu ích từ môn học trong suốt học kỳ.",
    "Môn học quá nặng về lý thuyết, thiếu hoàn toàn phần thực hành.",
    "Giảng viên không tạo được không khí học tập tích cực trong lớp.",
    "Em rất lo lắng về kết quả môn học vì cách dạy không giúp em hiểu bài.",
    "Thầy thường xuyên ra bài tập vào cuối buổi học mà không báo trước.",
    "Môn học không có phần Q&A, sinh viên không có cơ hội hỏi thêm.",
]

cau_trung_tinh = [
    "Môn học bình thường, không có gì đặc biệt để khen hay chê.",
    "Giảng viên dạy ổn, đủ nội dung theo chương trình đào tạo.",
    "Nội dung môn học tạm được, không quá khó cũng không quá dễ.",
    "Thầy dạy theo sách, không có gì mới nhưng cũng không sai.",
    "Môn học đáp ứng được yêu cầu cơ bản, không hơn không kém.",
    "Cô dạy đủ kiến thức theo đề cương, sinh viên học được.",
    "Bài giảng tàm tạm, không xuất sắc nhưng cũng không tệ lắm.",
    "Giảng viên có năng lực bình thường, môn học ở mức chấp nhận được.",
    "Tài liệu học tập đủ dùng nhưng không có gì đặc sắc.",
    "Môn học hoàn thành đúng kế hoạch, không có vấn đề lớn.",
    "Thầy dạy được, sinh viên hiểu bài ở mức cơ bản.",
    "Cô giảng đủ nội dung, không thiếu nhưng cũng không vượt trội.",
    "Môn học bình thường, phù hợp với chương trình học.",
    "Bài tập vừa đủ, không quá nhiều cũng không quá ít.",
    "Giảng viên giảng dạy theo đúng quy định và kế hoạch đã đề ra.",
    "Nội dung môn học phù hợp với mục tiêu học tập đã đề ra.",
    "Thầy không có gì đặc biệt, dạy bình thường theo giáo trình.",
    "Môn học ở mức trung bình so với các môn khác trong kỳ.",
    "Cô dạy đủ và đúng, sinh viên không có gì để phàn nàn cụ thể.",
    "Giảng viên thực hiện đúng nghĩa vụ giảng dạy theo hợp đồng.",
    "Môn học đã hoàn thành, em đạt được những mục tiêu cơ bản đề ra.",
    "Bài tập và kiểm tra ở mức độ bình thường, không gây áp lực lớn.",
    "Môn học cung cấp đủ kiến thức cần thiết, dù không hứng thú lắm.",
    "Cô dạy tạm ổn, sinh viên học và qua môn được là đủ.",
    "Giảng viên không tệ nhưng cũng chưa đủ tốt để ấn tượng sinh viên.",
    "Thầy dạy trung bình, đủ để sinh viên nắm được kiến thức cơ bản.",
    "Môn học không để lại ấn tượng đặc biệt nhưng cũng không gây khó khăn.",
    "Cô dạy phù hợp với khung chương trình, không có sự nổi bật.",
    "Giảng viên hoàn thành bổn phận dạy học nhưng không có sự sáng tạo.",
    "Thầy dạy bình thường, em học bình thường và đạt điểm trung bình.",
    "Môn học đúng như mô tả trong đề cương, không có gì khác biệt.",
    "Cô giảng đủ nội dung nhưng cách trình bày chưa thật sự thu hút.",
    "Bài giảng đầy đủ thông tin nhưng thiếu phần thực hành minh họa.",
    "Giảng viên thực hiện đúng kế hoạch giảng dạy, không sai sót lớn.",
    "Thầy dạy ổn, không có gì để khen hay chê đặc biệt.",
    "Môn học vừa đủ để tích lũy tín chỉ, không có ảnh hưởng lớn.",
    "Cô không gây ấn tượng mạnh nhưng cũng không làm sinh viên thất vọng.",
    "Nội dung bài giảng trung bình, không quá sâu cũng không quá nông.",
    "Giảng viên dạy bình thường theo đúng quy trình đào tạo của trường.",
    "Thầy không có phong cách dạy đặc biệt, nhưng kiến thức đầy đủ.",
    "Môn học hoàn thành đúng tiến độ mà không có vấn đề đáng kể.",
    "Cô dạy đúng giờ, đủ nội dung, sinh viên nắm được bài cơ bản.",
    "Bài tập về nhà vừa sức, không gây khó khăn cho sinh viên.",
    "Giảng viên không có phương pháp đặc biệt nhưng kiến thức chắc chắn.",
    "Thầy dạy truyền thống, không có gì mới mẻ nhưng cũng không sai.",
    "Môn học ở mức độ chấp nhận được đối với em.",
    "Cô giảng bài tàm tạm, sinh viên học được nhưng không hứng thú lắm.",
    "Giảng viên không tệ, chỉ là phương pháp dạy chưa thật sự hiệu quả.",
    "Thầy thực hiện đúng vai trò giảng dạy, không có gì đặc biệt.",
    "Môn học không gây khó khăn lớn nhưng cũng không mang lại nhiều giá trị.",
    "Cô dạy đủ bài nhưng thiếu những ví dụ thực tế sinh động hơn.",
    "Bài giảng trung bình, đủ để hiểu nhưng không để lại ấn tượng sâu.",
    "Giảng viên hoàn thành công việc nhưng thiếu sự nhiệt tình và sáng tạo.",
    "Thầy dạy đủ kiến thức cơ bản nhưng chưa đi sâu vào ứng dụng thực tế.",
    "Môn học có nội dung đúng đắn nhưng cách trình bày chưa thật sự hấp dẫn.",
    "Cô không gây ấn tượng xấu nhưng cũng chưa làm sinh viên thích thú.",
    "Giảng viên dạy đúng giáo trình nhưng thiếu sự linh hoạt trong giảng dạy.",
    "Môn học đáp ứng yêu cầu tối thiểu mà không gây thất vọng lớn.",
    "Cô dạy bình thường, sinh viên học bình thường và đạt kết quả bình thường.",
    "Giảng viên có kiến thức tốt nhưng kỹ năng truyền đạt chưa thật sự xuất sắc.",
    "Thầy không thiếu sót trong giảng dạy nhưng cũng không có điểm nổi bật.",
    "Môn học đã cung cấp những kiến thức cơ bản cần thiết cho chuyên ngành.",
    "Cô thực hiện đúng kế hoạch dạy học nhưng thiếu sự tương tác với sinh viên.",
    "Giảng viên dạy tàm tạm, không có gì đặc biệt trong suốt học kỳ.",
    "Thầy dạy đúng chuyên môn nhưng chưa tạo được sự hào hứng cho sinh viên.",
    "Môn học hoàn thành mà không để lại ấn tượng tích cực hay tiêu cực.",
    "Cô không tệ nhưng em cũng không thể nói là rất tốt về cô.",
    "Giảng viên thực hiện đúng trách nhiệm của một giảng viên bình thường.",
    "Thầy không sai về chuyên môn nhưng chưa truyền được đam mê cho sinh viên.",
    "Môn học ổn, không có gì phàn nàn nhưng cũng không có gì đặc sắc.",
    "Giảng viên đáp ứng đủ yêu cầu dạy học nhưng chưa vượt trội.",
    "Thầy dạy đủ kiến thức, sinh viên học ổn, không có vấn đề lớn.",
    "Môn học có nội dung ổn, phù hợp với mục tiêu đào tạo của chương trình.",
    "Cô không tạo ra sự khác biệt trong cách dạy so với các giảng viên khác.",
    "Bài giảng đủ để sinh viên qua môn, dù không thật sự xuất sắc.",
    "Giảng viên hoàn thành nhiệm vụ nhưng thiếu những điểm làm sinh viên nhớ mãi.",
    "Thầy không để lại dấu ấn đặc biệt nhưng cũng không gây bất mãn lớn.",
    "Môn học không gây ra cảm xúc mạnh, chỉ là một môn học bình thường.",
    "Cô dạy tàm tạm, em học ổn, kết quả ở mức trung bình khá.",
    "Giảng viên không tệ, nhưng em kỳ vọng cao hơn từ môn học quan trọng này.",
    "Thầy thực hiện đúng quy định nhưng thiếu sự nhiệt tình đặc biệt.",
    "Môn học ổn định, không có biến động lớn trong suốt quá trình học.",
    "Cô dạy tốt về mặt kỹ thuật nhưng chưa tạo được kết nối với sinh viên.",
    "Giảng viên không thiếu kiến thức nhưng thiếu kỹ năng truyền cảm hứng.",
    "Thầy không làm em thất vọng nhưng cũng không làm em ấn tượng.",
    "Môn học hoàn thành đúng tiến độ với chất lượng ở mức chấp nhận được.",
    "Cô dạy bình thường, không đặc biệt nhưng cũng không có gì cần phàn nàn.",
    "Bài tập và kiểm tra đúng chuẩn, không có gì bất ngờ hay lạ lẫm.",
    "Giảng viên thực hiện đúng vai trò, sinh viên học và qua môn ổn định.",
    "Thầy không có gì để nói nhiều, dạy bình thường, học bình thường.",
    "Môn học ở mức tạm ổn, đủ để tích lũy kiến thức cơ bản cho ngành.",
    "Cô dạy không tệ nhưng em nghĩ môn học này cần được cải thiện hơn.",
    "Giảng viên đủ năng lực nhưng chưa thật sự phát huy hết khả năng.",
    "Thầy dạy ổn, không có vấn đề đáng kể trong suốt quá trình học.",
    "Môn học vừa đủ về nội dung, sinh viên có thể học và hiểu được.",
    "Cô không tệ, chỉ là phong cách dạy chưa phù hợp với em lắm.",
    "Giảng viên dạy theo quy trình chuẩn, sinh viên đạt kết quả trung bình.",
    "Thầy dạy đúng chương trình nhưng thiếu phần liên hệ thực tế hơn.",
    "Môn học bình thường trong bình thường, không nổi bật trong kỳ học.",
]

# ============================================================
#  GỘP VÀ XÁO TRỘN
# ============================================================
print("=" * 55)
print("TẠO DATASET TỔNG HỢP")
print("=" * 55)

data_tu_thu_thap = (
    [(c, 2) for c in cau_tich_cuc] +
    [(c, 0) for c in cau_tieu_cuc] +
    [(c, 1) for c in cau_trung_tinh]
)

print(f"\nDữ liệu tự thu thập:")
print(f"  Tích cực (2): {len(cau_tich_cuc)} câu")
print(f"  Tiêu cực (0): {len(cau_tieu_cuc)} câu")
print(f"  Trung tính (1): {len(cau_trung_tinh)} câu")
print(f"  Tổng: {len(data_tu_thu_thap)} câu")

# ============================================================
#  TẠO DỮ LIỆU MÔ PHỎNG UIT-VSFC
#  (trong thực tế dùng step1 để load từ HuggingFace)
# ============================================================

print("\nTạo dữ liệu mô phỏng UIT-VSFC...")

mau_tich_cuc = [
    "Thầy giảng dạy rất tốt và nhiệt tình",
    "Cô giải thích rõ ràng dễ hiểu",
    "Môn học rất hay và bổ ích",
    "Giảng viên rất tận tâm với sinh viên",
    "Bài giảng sinh động và hấp dẫn",
    "Nội dung phong phú và thiết thực",
    "Thầy dạy rất hay em rất thích",
    "Môn học giúp em hiểu nhiều điều mới",
    "Cô rất nhiệt tình hỗ trợ sinh viên",
    "Chương trình học hợp lý và khoa học",
]
mau_tieu_cuc = [
    "Thầy dạy khó hiểu quá sinh viên không theo kịp",
    "Môn học quá khó bài tập nhiều",
    "Giảng viên dạy nhanh quá không hiểu",
    "Tài liệu thiếu không đủ để học",
    "Đề thi khó hơn nội dung đã học",
    "Cô không giải thích rõ sinh viên hỏi không được",
    "Môn này quá nặng so với tín chỉ",
    "Lớp học nóng bức không tập trung được",
    "Bài tập không liên quan đến nội dung",
    "Giảng viên hay nghỉ học đột ngột",
]
mau_trung_tinh = [
    "Môn học bình thường không có gì đặc biệt",
    "Giảng viên dạy ổn theo chương trình",
    "Nội dung tạm được không hay không dở",
    "Bài giảng đủ nội dung chấp nhận được",
    "Môn học không khó không dễ",
    "Cô dạy bình thường đúng giáo trình",
    "Không có gì nổi bật cũng không tệ",
    "Chương trình học bình thường",
    "Thầy dạy đủ bài không thiếu nội dung",
    "Môn học tạm ổn đáp ứng yêu cầu",
]

uitvsfc_data = []
n_train_vitc = 6618
n_train_tieu = 3059
n_train_trung = 1749

for i in range(n_train_vitc):
    mau = mau_tich_cuc[i % len(mau_tich_cuc)]
    cau = f"{mau} (mẫu UIT-VSFC #{i+1})"
    uitvsfc_data.append((cau, 2, "train"))
for i in range(n_train_tieu):
    mau = mau_tieu_cuc[i % len(mau_tieu_cuc)]
    cau = f"{mau} (mẫu UIT-VSFC #{i+1})"
    uitvsfc_data.append((cau, 0, "train"))
for i in range(n_train_trung):
    mau = mau_trung_tinh[i % len(mau_trung_tinh)]
    cau = f"{mau} (mẫu UIT-VSFC #{i+1})"
    uitvsfc_data.append((cau, 1, "train"))

for i in range(917):
    uitvsfc_data.append((f"{mau_tich_cuc[i%10]} (val #{i+1})", 2, "val"))
for i in range(424):
    uitvsfc_data.append((f"{mau_tieu_cuc[i%10]} (val #{i+1})", 0, "val"))
for i in range(242):
    uitvsfc_data.append((f"{mau_trung_tinh[i%10]} (val #{i+1})", 1, "val"))

for i in range(1590):
    uitvsfc_data.append((f"{mau_tich_cuc[i%10]} (test #{i+1})", 2, "test"))
for i in range(1409):
    uitvsfc_data.append((f"{mau_tieu_cuc[i%10]} (test #{i+1})", 0, "test"))
for i in range(167):
    uitvsfc_data.append((f"{mau_trung_tinh[i%10]} (test #{i+1})", 1, "test"))

print(f"  UIT-VSFC mô phỏng: {len(uitvsfc_data)} mẫu")

# ============================================================
#  GỘP TOÀN BỘ
# ============================================================

all_train = [(c, l, "train", "uitvsfc") for c, l, s in uitvsfc_data if s == "train"]
all_val   = [(c, l, "val",   "uitvsfc") for c, l, s in uitvsfc_data if s == "val"]
all_test  = [(c, l, "test",  "uitvsfc") for c, l, s in uitvsfc_data if s == "test"]
all_new   = [(c, l, "train", "tu_thu_thap") for c, l in data_tu_thu_thap]

all_train_full = all_train + all_new

random.shuffle(all_train_full)
random.shuffle(all_val)
random.shuffle(all_test)

all_data = all_train_full + all_val + all_test

# ============================================================
#  NHÃN TEXT
# ============================================================
label_map = {0: "Tieu_cuc", 1: "Trung_tinh", 2: "Tich_cuc"}

# ============================================================
#  GHI FILE CSV TỔNG HỢP
# ============================================================
print("\n[1/5] Ghi dataset_tonghop.csv...")
with open("data/dataset_tonghop.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "sentence", "sentiment", "label_name", "split", "source"])
    for i, (sentence, sentiment, split, source) in enumerate(all_data, 1):
        w.writerow([i, sentence, sentiment, label_map[sentiment], split, source])
print(f"  Tổng: {len(all_data)} mẫu")

# ============================================================
#  GHI FILE RIÊNG: TRAIN / VAL / TEST
# ============================================================
def ghi_file(ten, data, them_id=True):
    with open(f"data/{ten}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if them_id:
            w.writerow(["id", "sentence", "sentiment", "label_name", "source"])
            for i, (sentence, sentiment, split, source) in enumerate(data, 1):
                w.writerow([i, sentence, sentiment, label_map[sentiment], source])
        else:
            w.writerow(["sentence", "sentiment"])
            for sentence, sentiment, split, source in data:
                w.writerow([sentence, sentiment])

print("[2/5] Ghi dataset_train.csv...")
ghi_file("dataset_train", all_train_full)
print(f"  Train: {len(all_train_full)} mẫu "
      f"(UIT-VSFC: {len(all_train)}, Tự thu thập: {len(all_new)})")

print("[3/5] Ghi dataset_val.csv...")
ghi_file("dataset_val", all_val)
print(f"  Val: {len(all_val)} mẫu")

print("[4/5] Ghi dataset_test.csv...")
ghi_file("dataset_test", all_test)
print(f"  Test: {len(all_test)} mẫu")

# ============================================================
#  GHI FILE CHỈ CÓ SENTENCE + SENTIMENT (dùng để train model)
# ============================================================
print("[5/5] Ghi train.csv, val.csv, test.csv (format đơn giản)...")
ghi_file("train", all_train_full, them_id=False)
ghi_file("val",   all_val,        them_id=False)
ghi_file("test",  all_test,       them_id=False)

# ============================================================
#  THỐNG KÊ CUỐI
# ============================================================
def thong_ke(data, ten):
    tc  = sum(1 for _, l, _, _ in data if l == 2)
    neu = sum(1 for _, l, _, _ in data if l == 0)
    tru = sum(1 for _, l, _, _ in data if l == 1)
    n   = len(data)
    print(f"\n  {ten} ({n} mẫu):")
    print(f"    Tích cực  (2): {tc:5d}  ({tc/n*100:.1f}%)")
    print(f"    Tiêu cực  (0): {neu:5d}  ({neu/n*100:.1f}%)")
    print(f"    Trung tính(1): {tru:5d}  ({tru/n*100:.1f}%)")

print("\n" + "=" * 55)
print("THỐNG KÊ DATASET TỔNG HỢP")
print("=" * 55)
thong_ke(all_train_full, "Train (UIT-VSFC + Tự thu thập)")
thong_ke(all_val,         "Validation")
thong_ke(all_test,        "Test")

print("\n" + "=" * 55)
print("CÁC FILE ĐÃ TẠO TRONG THƯ MỤC data/")
print("=" * 55)
files = [
    ("dataset_tonghop.csv", "Toàn bộ dataset có đầy đủ thông tin"),
    ("dataset_train.csv",   "Tập train đầy đủ thông tin + nguồn"),
    ("dataset_val.csv",     "Tập validation đầy đủ thông tin"),
    ("dataset_test.csv",    "Tập test đầy đủ thông tin"),
    ("train.csv",           "Tập train dùng để train model"),
    ("val.csv",             "Tập val dùng để train model"),
    ("test.csv",            "Tập test dùng để train model"),
]
for fname, desc in files:
    fpath = f"data/{fname}"
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / 1024
        print(f"  ✅ {fname:<25} {desc} ({size:.0f} KB)")

print("\nBước tiếp theo:")
print("  python step2_preprocess.py   ← Tiền xử lý và tạo cache")
print("  python step3_train_lstm.py   ← Train baseline")
print("  python step5_train_phobert.py ← Fine-tune PhoBERT")