# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import sys
import unittest

from dashboard.api import api_auth
from dashboard.common import testing_common
from dashboard.pinpoint import test


class AuthTest(test.TestCase):

  def _ValidParams(self):
    builder_name = 'Mac Builder'
    change = '{"commits": [{"repository": "chromium", "git_hash": "git hash"}]}'
    target = 'telemetry_perf_tests'
    isolate_server = 'https://isolate.server'
    isolate_hash = 'a0c28d99182661887feac644317c94fa18eccbbb'

    params = {
        'builder_name': builder_name,
        'change': change,
        'isolate_server': isolate_server,
        'isolate_map': json.dumps({target: isolate_hash}),
    }
    return params

  def testPost_Anonymous_Allowlisted_Succeeds(self):
    testing_common.SetIpAllowlist(['remote_ip'])
    self.SetCurrentUserOAuth(None)

    self.Post('/api/isolate', self._ValidParams(), status=200)

  def testPost_Anonymous_NotAllowlisted_Fails(self):
    testing_common.SetIpAllowlist(['invalid'])
    self.SetCurrentUserOAuth(None)

    self.Post('/api/isolate', self._ValidParams(), status=401)

  def testPost_Internal_Oauth_Succeeds(self):
    testing_common.SetIpAllowlist(['invalid'])
    self.SetCurrentUserOAuth(testing_common.INTERNAL_USER)
    self.SetCurrentClientIdOAuth(api_auth.OAUTH_CLIENT_ID_ALLOWLIST[0])

    self.Post('/api/isolate', self._ValidParams(), status=200)

  def testPost_External_Oauth_Fails(self):
    testing_common.SetIpAllowlist(['invalid'])
    self.SetCurrentUserOAuth(testing_common.EXTERNAL_USER)
    self.SetCurrentClientIdOAuth(api_auth.OAUTH_CLIENT_ID_ALLOWLIST[0])

    self.Post('/api/isolate', self._ValidParams(), status=403)


class FunctionalityTest(test.TestCase):

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testPostAndGet(self):
    testing_common.SetIpAllowlist(['remote_ip'])

    builder_name = 'Mac Builder'
    change = '{"commits": [{"repository": "chromium", "git_hash": "git hash"}]}'
    target = 'telemetry_perf_tests'
    isolate_server = 'https://isolate.server'
    isolate_hash = 'a0c28d99182661887feac644317c94fa18eccbbb'

    params = {
        'builder_name': builder_name,
        'change': change,
        'isolate_server': isolate_server,
        'isolate_map': json.dumps({target: isolate_hash}),
    }
    self.testapp.post('/api/isolate', params, status=200)

    params = {
        'builder_name': builder_name,
        'change': change,
        'target': target,
    }
    response = self.testapp.get('/api/isolate', params, status=200)
    expected_body = json.dumps({
        'isolate_server': isolate_server,
        'isolate_hash': isolate_hash
    })
    self.assertEqual(response.normal_body, expected_body)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testGetUnknownIsolate(self):
    params = {
        'builder_name':
            'Mac Builder',
        'change':
            '{"commits": [{"repository": "chromium", "git_hash": "hash"}]}',
        'target':
            'not a real target',
    }
    self.testapp.get('/api/isolate', params, status=404)

  def testPostPermissionDenied(self):
    testing_common.SetIpAllowlist([])
    self.testapp.post('/api/isolate', status=401)


class ParameterValidationTest(test.TestCase):

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testExtraParameter(self):
    params = {
        'builder_name':
            'Mac Builder',
        'change':
            '{"commits": [{"repository": "chromium", "git_hash": "hash"}]}',
        'target':
            'telemetry_perf_tests',
        'extra_parameter':
            '',
    }
    self.testapp.get('/api/isolate', params, status=400)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testMissingParameter(self):
    params = {
        'builder_name':
            'Mac Builder',
        'change':
            '{"commits": [{"repository": "chromium", "git_hash": "hash"}]}',
    }
    self.testapp.get('/api/isolate', params, status=400)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testEmptyParameter(self):
    params = {
        'builder_name':
            'Mac Builder',
        'change':
            '{"commits": [{"repository": "chromium", "git_hash": "hash"}]}',
        'target':
            '',
    }
    self.testapp.get('/api/isolate', params, status=400)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testBadJson(self):
    params = {
        'builder_name': 'Mac Builder',
        'change': '',
        'target': 'telemetry_perf_tests',
    }
    self.testapp.get('/api/isolate', params, status=400)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testBadChange(self):
    params = {
        'builder_name': 'Mac Builder',
        'change': '{"commits": [{}]}',
        'target': 'telemetry_perf_tests',
    }
    self.testapp.get('/api/isolate', params, status=400)

  @unittest.skipIf(sys.version_info.major == 3,
                   'Skipping get handler of api/isolate for python 3.')
  def testGetInvalidChangeBecauseOfUnknownRepository(self):
    params = {
        'builder_name': 'Mac Builder',
        'change': '{"commits": [{"repository": "foo", "git_hash": "hash"}]}',
        'target': 'telemetry_perf_tests',
    }
    self.testapp.get('/api/isolate', params, status=400)

  def testPostInvalidChangeBecauseOfUnknownRepository(self):
    testing_common.SetIpAllowlist(['remote_ip'])

    params = {
        'builder_name': 'Mac Builder',
        'change': '{"commits": [{"repository": "foo", "git_hash": "hash"}]}',
        'isolate_map': '{"telemetry_perf_tests": "a0c28d9"}',
    }
    self.testapp.post('/api/isolate', params, status=400)
