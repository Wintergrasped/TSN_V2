-- --------------------------------------------------------
-- Host:                         51.81.202.9
-- Server version:               10.11.14-MariaDB-0+deb12u2 - Debian 12
-- Server OS:                    debian-linux-gnu
-- HeidiSQL Version:             12.12.0.7122
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Dumping data for table repeater.gpu_utilization_samples: ~13 rows (approximately)
INSERT INTO `gpu_utilization_samples` (`utilization_pct`, `sample_source`, `is_saturated`, `notes`, `id`, `created_at`, `updated_at`) VALUES
	(0, 'nvidia-smi', 0, NULL, '24f30484-4f39-41c1-9fe8-f45bf2adc786', '2026-01-28 16:47:35', '2026-01-28 16:47:35'),
	(0, 'nvidia-smi', 0, NULL, '2cd09858-f1d9-4961-817d-0827827cb26c', '2026-01-28 17:18:01', '2026-01-28 17:18:01'),
	(0, 'nvidia-smi', 0, NULL, '4a5e7b49-1d31-443b-ab43-1c4f5ddac981', '2026-01-28 07:03:47', '2026-01-28 07:03:47'),
	(0, 'nvidia-smi', 0, NULL, '4e55517d-741d-4213-8084-23e0be62a37e', '2026-01-28 14:14:16', '2026-01-28 14:14:16'),
	(0, 'nvidia-smi', 0, NULL, '549db18e-5404-4478-bc11-1dcbbbf09cf9', '2026-01-28 14:13:59', '2026-01-28 14:13:59'),
	(0, 'nvidia-smi', 0, NULL, '558c699c-e48a-472d-9313-0d3b39316a96', '2026-01-28 14:13:59', '2026-01-28 14:13:59'),
	(97, 'nvidia-smi', 1, NULL, '82a41b15-04c2-4b71-8ac6-3a3f249259d5', '2026-01-28 06:17:43', '2026-01-28 06:17:43'),
	(0, 'nvidia-smi', 0, NULL, '849daceb-a4a6-4d58-9aff-8df208babb6b', '2026-01-28 17:18:01', '2026-01-28 17:18:01'),
	(0, 'nvidia-smi', 0, NULL, '905a863a-bbbf-4fb1-a462-fc1328813ddf', '2026-01-28 16:47:35', '2026-01-28 16:47:35'),
	(0, 'nvidia-smi', 0, NULL, 'b0616604-1ca2-463d-b3ce-b965c6505c8a', '2026-01-28 06:17:24', '2026-01-28 06:17:24'),
	(0, 'nvidia-smi', 0, NULL, 'b2d453c7-674a-4013-ba20-9dc96ff69da6', '2026-01-28 14:14:16', '2026-01-28 14:14:16'),
	(0, 'nvidia-smi', 0, NULL, 'db174930-6395-4a1f-bbd5-5d5fc01f4e18', '2026-01-28 07:04:02', '2026-01-28 07:04:02'),
	(0, 'nvidia-smi', 0, NULL, 'e1909e73-8196-4ebb-b0c7-ab0d46eddd5c', '2026-01-28 07:03:47', '2026-01-28 07:03:47');

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
