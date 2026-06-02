from unittest import mock

import pytest

from gcsfs.extended_gcsfs import ExtendedGcsFileSystem


@pytest.mark.asyncio
async def test_rm_mixed_list_hns_and_non_hns_unit():
    fs = ExtendedGcsFileSystem()
    fs._is_bucket_hns_enabled = mock.AsyncMock(side_effect=lambda b: b == "hns-bucket")
    fs._expand_path_with_details = mock.AsyncMock(
        return_value=[{"name": "hns-bucket/file1", "type": "file"}]
    )
    fs._perform_rm = mock.AsyncMock()

    with mock.patch(
        "gcsfs.core.GCSFileSystem._rm", new=mock.AsyncMock()
    ) as mock_super_rm:
        paths = ["hns-bucket/file1", "flat-bucket/file2", "hns-bucket/file3"]
        await fs._rm(paths)

        mock_super_rm.assert_called_once_with(
            ["flat-bucket/file2"], recursive=False, maxdepth=None, batchsize=20
        )
        fs._perform_rm.assert_called_once_with(
            ["hns-bucket/file1"],
            [],
            ["hns-bucket/file1", "hns-bucket/file3"],
            batchsize=20,
        )
