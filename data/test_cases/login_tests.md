# Test Cases: Login (covers US-101)

## TC-101-01: Valid login
**Steps:** Enter valid email + correct password, click Login.
**Expected:** User is redirected to dashboard.

## TC-101-02: Invalid password
**Steps:** Enter valid email + wrong password.
**Expected:** Error "Invalid email or password." shown.

## TC-101-03: Account lockout after 5 failed attempts
**Steps:** Enter wrong password 5 times within 10 minutes.
**Expected:** Account locks for 15 minutes; lockout message displayed on 5th attempt.

## TC-101-04: Lockout resets after 15 minutes
**Steps:** Trigger lockout, wait 15 minutes, attempt login with correct password.
**Expected:** Login succeeds.

## TC-101-05: Password field masking
**Steps:** Type password into password field.
**Expected:** Characters are masked (dots/asterisks), not shown in plaintext.