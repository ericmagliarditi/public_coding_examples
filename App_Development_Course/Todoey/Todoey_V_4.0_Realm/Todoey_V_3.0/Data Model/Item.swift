//
//  Item.swift
//  Todoey_V_3.0
//
//  Created by Eric Magliarditi on 6/24/19.
//  Copyright © 2019 Eric Magliarditi. All rights reserved.
//

import Foundation
import RealmSwift

class Item: Object {
    @objc dynamic var title: String = ""
    @objc dynamic var done: Bool = false
}
